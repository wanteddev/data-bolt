"""RAG lookups for schema and glossary context."""

from __future__ import annotations

import logging
import os
from typing import Any

from .llm_client import _laas_post
from .llm_config import LAAS_RAG_GLOSSARY_COLLECTION, LAAS_RAG_SCHEMA_COLLECTION

logger = logging.getLogger(__name__)


def _vector_search(
    *,
    collection: str,
    text: str,
    limit: int,
    min_score: float,
) -> list[dict[str, Any]]:
    payload = {
        "text": text,
        "limit": limit,
        "offset": 0,
        "with_metadata": True,
        "with_vector": False,
        "min_score": min_score,
    }
    try:
        resp = _laas_post(f"/api/document/{collection}/similar/text", payload, timeout=30.0)
        return resp if isinstance(resp, list) else []
    except Exception as e:
        logger.warning(
            "rag.vector_search_failed",
            extra={"collection": collection, "error": str(e)},
        )
        return []


def _join_doc_texts(docs: list[dict[str, Any]]) -> str:
    texts: list[str] = []
    for doc in docs:
        try:
            text = doc.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        except Exception:
            continue
    return "\n".join(texts)


def _collect_rag_context(question: str) -> dict[str, Any]:
    schema_collection = os.getenv("LAAS_RAG_SCHEMA_COLLECTION", LAAS_RAG_SCHEMA_COLLECTION)
    glossary_collection = os.getenv("LAAS_RAG_GLOSSARY_COLLECTION", LAAS_RAG_GLOSSARY_COLLECTION)
    schema_limit = int(os.getenv("LAAS_RAG_SCHEMA_LIMIT", "64"))
    glossary_limit = int(os.getenv("LAAS_RAG_GLOSSARY_LIMIT", "5"))
    schema_min_score = float(os.getenv("LAAS_RAG_SCHEMA_MIN_SCORE", "0.5"))
    glossary_min_score = float(os.getenv("LAAS_RAG_GLOSSARY_MIN_SCORE", "0.5"))

    schema_docs = _vector_search(
        collection=schema_collection,
        text=question,
        limit=schema_limit,
        min_score=schema_min_score,
    )
    glossary_docs = _vector_search(
        collection=glossary_collection,
        text=question,
        limit=glossary_limit,
        min_score=glossary_min_score,
    )

    table_info = _join_doc_texts(schema_docs)
    glossary_info = _join_doc_texts(glossary_docs)

    return {
        "table_info": table_info,
        "glossary_info": glossary_info,
        "meta": {
            "attempted": True,
            "schema_collection": schema_collection,
            "glossary_collection": glossary_collection,
            "schema_docs": len(schema_docs),
            "glossary_docs": len(glossary_docs),
            "table_info_chars": len(table_info),
            "glossary_info_chars": len(glossary_info),
        },
    }
