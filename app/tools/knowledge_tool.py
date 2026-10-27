"""知识检索工具 - 从向量数据库中检索相关信息"""

import time
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.services.vector_store_manager import vector_store_manager
from app.services.rag_metrics import rag_metrics


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题
    
    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。
    
    Args:
        query: 用户的问题或查询
        
    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    started_at = time.perf_counter()
    try:
        logger.info(f"知识检索工具被调用: query='{query}'")
        
        # 从向量存储中检索相关文档
        # 直接调用带分数接口，保留 Milvus 原始分数用于检索质量分析。
        scored_docs = vector_store_manager.similarity_search_with_score(
            query, k=config.rag_top_k
        )
        docs = [document for document, _score in scored_docs]
        scores = [float(score) for _document, score in scored_docs]
        latency_ms = (time.perf_counter() - started_at) * 1000
        try:
            rag_metrics.record_retrieval(
                query=query,
                top_k=config.rag_top_k,
                documents=docs,
                scores=scores,
                latency_ms=latency_ms,
            )
        except Exception as metrics_error:
            # 指标写入不能影响已经成功的知识检索。
            logger.warning(f"RAG 检索指标记录失败: {metrics_error}")
        
        if not docs:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []
        
        # 格式化文档为上下文
        context = format_docs(docs)
        
        logger.info(f"检索到 {len(docs)} 个相关文档")
        return context, docs
        
    except Exception as e:
        # 记录失败请求，避免成功率被错误请求静默抬高。
        try:
            rag_metrics.record_retrieval(
                query=query,
                top_k=config.rag_top_k,
                documents=[],
                scores=[],
                latency_ms=(time.perf_counter() - started_at) * 1000,
                error=e,
            )
        except Exception:
            # 指标本身不能覆盖原始检索异常。
            pass
        logger.error(f"知识检索工具调用失败: {e}")
        return f"检索知识时发生错误: {str(e)}", []


def format_docs(docs: List[Document]) -> str:
    """
    格式化文档列表为上下文文本
    
    Args:
        docs: 文档列表
        
    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []
    
    for i, doc in enumerate(docs, 1):
        # 提取元数据
        metadata = doc.metadata
        source = metadata.get("_file_name", "未知来源")
        
        # 提取标题信息 (如果有)
        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])
        
        header_str = " > ".join(headers) if headers else ""
        
        # 构建格式化文本
        formatted = f"【参考资料 {i}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"
        
        formatted_parts.append(formatted)
    
    return "\n".join(formatted_parts)
