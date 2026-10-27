from pathlib import Path

from langchain_core.documents import Document

from app.services.rag_metrics import (
    RagMetricsCollector,
    RetrievalExample,
    document_key,
    evaluate_retrieval,
)


def _doc(key: str) -> Document:
    return Document(page_content=key, metadata={"chunk_id": key})


def test_evaluate_retrieval_metrics() -> None:
    documents = {
        "q1": [_doc("a"), _doc("b"), _doc("c")],
        "q2": [_doc("x"), _doc("y"), _doc("z")],
    }
    examples = [
        RetrievalExample("q1", frozenset({"b"})),
        RetrievalExample("q2", frozenset({"missing"})),
    ]

    result = evaluate_retrieval(examples, lambda query, _k: documents[query], k=3)

    assert result == {
        "sample_count": 2,
        "k": 3,
        "answerable_sample_count": 2,
        "no_answer_sample_count": 0,
        "hit_rate_at_k": 0.5,
        "recall_at_k": 0.5,
        "precision_at_k": 0.1667,
        "mrr_at_k": 0.25,
        "no_answer_accuracy_at_k": None,
        "false_positive_rate_at_k": None,
    }


def test_metrics_snapshot_and_jsonl(tmp_path: Path) -> None:
    collector = RagMetricsCollector(log_path=tmp_path / "metrics.jsonl")
    collector.record_retrieval(
        query="disk usage",
        top_k=3,
        documents=[_doc("a")],
        scores=[0.25],
        latency_ms=10,
    )
    collector.record_retrieval(
        query="unknown",
        top_k=3,
        documents=[],
        scores=[],
        latency_ms=30,
        error="backend unavailable",
    )

    snapshot = collector.snapshot()
    assert snapshot["window_events"] == 2
    assert snapshot["config"]["top_k"] == 3
    assert snapshot["successful_queries"] == 1
    assert snapshot["failed_queries"] == 1
    assert snapshot["success_rate"] == 0.5
    assert snapshot["avg_latency_ms"] == 10.0
    assert snapshot["p95_latency_ms"] == 10.0
    assert snapshot["avg_score"] == 0.25

    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "disk usage" not in lines[0]
    assert document_key(_doc("a")) == "a"


def test_evaluate_retrieval_accepts_source_labels() -> None:
    results = [
        Document(page_content="chunk 1", metadata={"_source": "/docs/ops.md"}),
        Document(page_content="chunk 2", metadata={"_source": "/docs/ops.md"}),
    ]

    result = evaluate_retrieval(
        [{"query": "ops", "relevant_sources": ["ops.md"]}],
        lambda _query, _k: results,
        k=2,
    )

    assert result["hit_rate_at_k"] == 1.0
    assert result["recall_at_k"] == 1.0


def test_evaluate_retrieval_no_answer_metrics() -> None:
    result = evaluate_retrieval(
        [
            {"query": "known", "relevant_sources": ["ops.md"]},
            {"query": "unknown", "expected_no_answer": True, "relevant_sources": []},
        ],
        lambda query, _k: (
            [Document(page_content="known", metadata={"_file_name": "ops.md"})]
            if query == "known"
            else []
        ),
        k=3,
    )

    assert result["answerable_sample_count"] == 1
    assert result["no_answer_sample_count"] == 1
    assert result["no_answer_accuracy_at_k"] == 1.0
    assert result["false_positive_rate_at_k"] == 0.0
