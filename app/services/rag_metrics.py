"""RAG 检索指标收集与离线评测。

运行时指标用于回答“系统现在表现如何”，离线评测用于回答“参数调整后是否变好”。
评测实现只依赖检索结果，不依赖具体向量数据库或大模型，便于在 CI 和脚本中复用。
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from langchain_core.documents import Document
from loguru import logger

from app.config import config


def document_key(document: Document) -> str:
    """返回可用于评测的稳定文档标识。

    向量库的主键未必会被 LangChain 返回，因此优先使用业务侧 metadata，
    最后使用内容哈希，确保同一份评测结果在不同存储实现中仍可比较。
    """

    metadata = document.metadata or {}
    for key in ("_vector_id", "id", "_id", "chunk_id"):
        value = metadata.get(key)
        if value is not None:
            return str(value)

    source = metadata.get("_source") or metadata.get("source") or metadata.get("_file_name") or ""
    content_hash = hashlib.sha1(document.page_content.encode("utf-8")).hexdigest()[:16]
    return f"{source}::{content_hash}"


@dataclass(frozen=True)
class RetrievalEvent:
    """一次检索调用的结构化记录。"""

    query_id: str
    query_hash: str
    top_k: int
    result_count: int
    latency_ms: float
    scores: list[float] = field(default_factory=list)
    document_keys: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    success: bool = True
    error: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IndexingEvent:
    """一次文件索引调用的结构化记录。"""

    source: str
    chunk_count: int
    total_chars: int
    avg_chunk_chars: float
    latency_ms: float
    success: bool = True
    error: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RagMetricsCollector:
    """线程安全的 RAG 运行时指标收集器。

    内存中保留最近 ``max_events`` 条事件，同时以 JSONL 追加写入磁盘，
    这样 API 可以快速查询当前进程统计，重启后也能保留原始事件用于分析。
    """

    def __init__(self, log_path: str | Path | None = None, max_events: int = 5000):
        self.log_path = Path(log_path or config.rag_metrics_log_path)
        self.max_events = max(1, max_events)
        self._events: list[RetrievalEvent] = []
        self._indexing_events: list[IndexingEvent] = []
        self._lock = threading.Lock()

    def record_retrieval(
        self,
        *,
        query: str,
        top_k: int,
        documents: Sequence[Document],
        latency_ms: float,
        scores: Sequence[float] | None = None,
        query_id: str | None = None,
        error: Exception | str | None = None,
    ) -> RetrievalEvent:
        """记录一次检索调用，并返回事件对象。"""

        sources = []
        keys = []
        for document in documents:
            keys.append(document_key(document))
            metadata = document.metadata or {}
            source = metadata.get("_file_name") or metadata.get("_source")
            if source:
                sources.append(str(source))

        normalized_scores: list[float] = []
        for score in scores or []:
            try:
                normalized_scores.append(float(score))
            except (TypeError, ValueError):
                continue

        event = RetrievalEvent(
            query_id=query_id or hashlib.sha1(f"{time.time_ns()}:{query}".encode()).hexdigest()[:16],
            query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            top_k=max(0, int(top_k)),
            result_count=len(documents),
            latency_ms=round(max(0.0, float(latency_ms)), 3),
            scores=normalized_scores,
            document_keys=keys,
            sources=sorted(set(sources)),
            success=error is None,
            error=str(error) if error is not None else None,
        )

        with self._lock:
            self._events.append(event)
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events :]
            self._append_event(event, event_type="retrieval")
        return event

    def _append_event(self, event: RetrievalEvent | IndexingEvent, *, event_type: str) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            payload = event.to_dict()
            payload["event_type"] = event_type
            with self.log_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as exc:
            # 指标不能影响用户的知识检索请求。
            logger.warning(f"RAG 指标写入失败: {exc}")

    def events(self) -> list[RetrievalEvent]:
        with self._lock:
            return list(self._events)

    def record_indexing(
        self,
        *,
        source: str,
        chunk_count: int,
        total_chars: int,
        latency_ms: float,
        error: Exception | str | None = None,
    ) -> IndexingEvent:
        """记录一次文件分片和向量化索引。"""

        count = max(0, int(chunk_count))
        chars = max(0, int(total_chars))
        event = IndexingEvent(
            source=source,
            chunk_count=count,
            total_chars=chars,
            avg_chunk_chars=round(chars / count, 3) if count else 0.0,
            latency_ms=round(max(0.0, float(latency_ms)), 3),
            success=error is None,
            error=str(error) if error is not None else None,
        )
        with self._lock:
            self._indexing_events.append(event)
            if len(self._indexing_events) > self.max_events:
                self._indexing_events = self._indexing_events[-self.max_events :]
            self._append_event(event, event_type="indexing")
        return event

    def snapshot(self) -> dict[str, Any]:
        """计算当前进程最近事件的聚合指标。"""

        events = self.events()
        total = len(events)
        successful = [event for event in events if event.success]
        latencies = sorted(event.latency_ms for event in successful)
        result_counts = [event.result_count for event in successful]
        score_values = [score for event in successful for score in event.scores]
        indexing_events = self._indexing_snapshot()

        def percentile(values: list[float], percentile_value: float) -> float | None:
            if not values:
                return None
            index = (len(values) - 1) * percentile_value
            lower = math.floor(index)
            upper = math.ceil(index)
            if lower == upper:
                return round(values[lower], 3)
            return round(values[lower] + (values[upper] - values[lower]) * (index - lower), 3)

        return {
            "config": {
                "top_k": config.rag_top_k,
                "chunk_max_size": config.chunk_max_size,
                "chunk_overlap": config.chunk_overlap,
                "embedding_model": config.dashscope_embedding_model,
            },
            "window_events": total,
            "successful_queries": len(successful),
            "failed_queries": total - len(successful),
            "success_rate": round(len(successful) / total, 4) if total else None,
            "empty_result_rate": (
                round(sum(event.result_count == 0 for event in successful) / len(successful), 4)
                if successful
                else None
            ),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "p50_latency_ms": percentile(latencies, 0.50),
            "p95_latency_ms": percentile(latencies, 0.95),
            "avg_result_count": round(sum(result_counts) / len(result_counts), 3)
            if result_counts
            else None,
            "avg_score": round(sum(score_values) / len(score_values), 6)
            if score_values
            else None,
            "indexing": indexing_events,
        }

    def _indexing_snapshot(self) -> dict[str, Any]:
        with self._lock:
            events = list(self._indexing_events)
        successful = [event for event in events if event.success]
        total_chunks = sum(event.chunk_count for event in successful)
        total_chars = sum(event.total_chars for event in successful)
        return {
            "files": len(successful),
            "failed_files": len(events) - len(successful),
            "chunks": total_chunks,
            "avg_chunk_chars": round(total_chars / total_chunks, 3) if total_chunks else None,
            "avg_indexing_latency_ms": round(
                sum(event.latency_ms for event in successful) / len(successful), 3
            )
            if successful
            else None,
        }

    def reset(self) -> None:
        """清空当前进程窗口；磁盘历史不会删除。"""

        with self._lock:
            self._events.clear()
            self._indexing_events.clear()


@dataclass(frozen=True)
class RetrievalExample:
    """离线检索评测样本。

    ``relevant_document_keys`` 可以使用 chunk ID，也可以使用文档来源路径/文件名；
    后者适合人工标注，不需要提前知道向量库生成的随机主键。
    """

    query: str
    relevant_document_keys: frozenset[str]
    expected_no_answer: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RetrievalExample":
        relevant = value.get(
            "relevant_document_keys",
            value.get("relevant_sources", value.get("relevant", [])),
        )
        expected_no_answer = bool(value.get("expected_no_answer", value.get("answerable") is False))
        return cls(
            query=str(value["query"]),
            relevant_document_keys=frozenset(map(str, relevant)),
            expected_no_answer=expected_no_answer,
        )


def evaluate_retrieval(
    examples: Iterable[RetrievalExample | Mapping[str, Any]],
    search: Callable[[str, int], Sequence[Document]],
    *,
    k: int = 3,
) -> dict[str, Any]:
    """对人工标注问题集计算 Hit@K、Recall@K、Precision@K、MRR@K。

    ``search`` 应返回按相关性降序排列的文档；生产环境可传入
    ``vector_store_manager.similarity_search``，测试环境可传入内存检索函数。
    """

    if k <= 0:
        raise ValueError("k 必须大于 0")

    normalized = [
        example if isinstance(example, RetrievalExample) else RetrievalExample.from_mapping(example)
        for example in examples
    ]
    if not normalized:
        return {"sample_count": 0, "k": k, "hit_rate_at_k": None, "recall_at_k": None,
                "precision_at_k": None, "mrr_at_k": None, "answerable_sample_count": 0,
                "no_answer_sample_count": 0, "no_answer_accuracy_at_k": None,
                "false_positive_rate_at_k": None}

    invalid_answerable = [
        example.query
        for example in normalized
        if not example.expected_no_answer and not example.relevant_document_keys
    ]
    if invalid_answerable:
        raise ValueError(
            "可回答样本至少需要一个 relevant_document_keys 或 relevant_sources: "
            + ", ".join(invalid_answerable[:3])
        )

    hits = recalls = precisions = reciprocal_ranks = 0.0
    answerable_examples = [example for example in normalized if not example.expected_no_answer]
    no_answer_examples = [example for example in normalized if example.expected_no_answer]

    for example in answerable_examples:
        retrieved = list(search(example.query, k))[:k]
        retrieved_keys = [document_key(document) for document in retrieved]
        retrieved_sources = [
            str(
                (document.metadata or {}).get("_source")
                or (document.metadata or {}).get("source")
                or (document.metadata or {}).get("_file_name")
                or ""
            )
            for document in retrieved
        ]
        relevant = example.relevant_document_keys
        matched_indexes = [
            index
            for index, (key, source) in enumerate(zip(retrieved_keys, retrieved_sources))
            if key in relevant or source in relevant or Path(source).name in relevant
        ]
        matched = [retrieved_keys[index] for index in matched_indexes]
        matched_labels = {
            label
            for label in relevant
            if any(
                label == retrieved_keys[index]
                or label == retrieved_sources[index]
                or label == Path(retrieved_sources[index]).name
                for index in matched_indexes
            )
        }
        hits += bool(matched)
        recalls += len(matched_labels) / len(relevant) if relevant else 0.0
        precisions += len(set(matched)) / k
        reciprocal_ranks += 1 / (matched_indexes[0] + 1) if matched_indexes else 0.0

    no_answer_correct = 0
    for example in no_answer_examples:
        retrieved = list(search(example.query, k))[:k]
        no_answer_correct += not retrieved

    answerable_count = len(answerable_examples)
    no_answer_count = len(no_answer_examples)
    return {
        "sample_count": len(normalized),
        "k": k,
        "answerable_sample_count": answerable_count,
        "no_answer_sample_count": no_answer_count,
        "hit_rate_at_k": round(hits / answerable_count, 4) if answerable_count else None,
        "recall_at_k": round(recalls / answerable_count, 4) if answerable_count else None,
        "precision_at_k": round(precisions / answerable_count, 4) if answerable_count else None,
        "mrr_at_k": round(reciprocal_ranks / answerable_count, 4) if answerable_count else None,
        "no_answer_accuracy_at_k": round(no_answer_correct / no_answer_count, 4)
        if no_answer_count
        else None,
        "false_positive_rate_at_k": round(1 - no_answer_correct / no_answer_count, 4)
        if no_answer_count
        else None,
    }


rag_metrics = RagMetricsCollector()
