#!/usr/bin/env python3
"""Run an offline retrieval evaluation against the configured Milvus collection.

Input JSON can be either a list of examples or ``{"examples": [...]}``.
Each answerable example contains ``query`` and one of ``relevant_sources``, ``relevant``
or ``relevant_document_keys``. No-answer examples set ``expected_no_answer`` to true.
Source labels may be a file name, source path, or chunk ID.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Support direct execution: ``python scripts/evaluate_rag.py ...``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.rag_metrics import RetrievalExample, evaluate_retrieval


def load_examples(path: Path) -> list[RetrievalExample]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("examples", [])
    if not isinstance(payload, list):
        raise ValueError("评测文件必须是数组，或包含 examples 数组的 JSON 对象")
    if any(not isinstance(item, dict) for item in payload):
        raise ValueError("评测集中的每条样本必须是 JSON 对象")
    return [RetrievalExample.from_mapping(item) for item in payload]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OpsPilot RAG retrieval metrics")
    parser.add_argument("dataset", type=Path, help="人工标注评测集 JSON 文件")
    parser.add_argument("-k", "--k", type=int, default=3, help="评测 Top-K，默认 3")
    args = parser.parse_args()

    examples = load_examples(args.dataset)
    # Delay the Milvus import so ``--help`` and dataset validation work offline.
    from app.services.vector_store_manager import vector_store_manager

    result = evaluate_retrieval(
        examples,
        lambda query, limit: vector_store_manager.similarity_search(query, k=limit),
        k=args.k,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
