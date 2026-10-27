"""RAG 指标查询与离线评测接口。"""

from typing import Any

from fastapi import APIRouter, HTTPException

from app.services.rag_metrics import RetrievalExample, evaluate_retrieval, rag_metrics
from app.services.vector_store_manager import vector_store_manager

router = APIRouter(prefix="/rag", tags=["RAG 指标"])


@router.get("/metrics")
async def get_rag_metrics() -> dict[str, Any]:
    """返回当前进程窗口内的检索运行时指标。"""

    return {"code": 200, "message": "success", "data": rag_metrics.snapshot()}


@router.post("/metrics/reset")
async def reset_rag_metrics() -> dict[str, Any]:
    """清空内存统计窗口，不删除 JSONL 历史记录。"""

    rag_metrics.reset()
    return {"code": 200, "message": "success", "data": rag_metrics.snapshot()}


@router.post("/evaluate")
async def evaluate_rag(payload: dict[str, Any]) -> dict[str, Any]:
    """使用标注集评估当前向量检索配置。

    请求示例：{"k": 3, "examples": [{"query": "磁盘告警", "relevant_sources": ["disk_high_usage.md"]}]}
    ``relevant_sources`` 可填写文件名、来源路径或 chunk ID；无答案问题使用
    ``expected_no_answer: true`` 和空的 ``relevant_sources``。
    """

    try:
        k = int(payload.get("k", 3))
        raw_examples = payload.get("examples", [])
        if not isinstance(raw_examples, list):
            raise ValueError("examples 必须是数组")
        examples = [RetrievalExample.from_mapping(item) for item in raw_examples]
        result = evaluate_retrieval(
            examples,
            lambda query, limit: vector_store_manager.similarity_search(query, k=limit),
            k=k,
        )
        return {"code": 200, "message": "success", "data": result}
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
