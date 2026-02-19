"""Framework-agnostic RAG lookup helpers for schema and glossary context."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any, Protocol, cast

import httpx

if TYPE_CHECKING:

    class SSMClient(Protocol):
        def get_parameter(self, *, Name: str, WithDecryption: bool = False) -> dict[str, Any]: ...


logger = logging.getLogger(__name__)

LAAS_DEFAULT_BASE_URL = "https://api-laas.wanted.co.kr"
LAAS_API_KEY_SSM_PARAM = "/DATA/PIPELINE/API_KEY/OPENAI"
LAAS_RAG_SCHEMA_COLLECTION = "RAG_DATA_CATALOG"
LAAS_RAG_GLOSSARY_COLLECTION = "RAG_GLOSSARY"


class _SSMParameterLoader:
    def __init__(self, cache_ttl: int = 300) -> None:
        self._cache: dict[str, tuple[float, str]] = {}
        self._cache_ttl = cache_ttl
        self._client: SSMClient | None = None

    def get_parameter(self, key: str, with_decryption: bool = False) -> str:
        cache_key = f"{key}:{with_decryption}"
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached:
            expires_at, value = cached
            if now < expires_at:
                return value

        if self._client is None:
            import boto3

            self._client = cast("SSMClient", boto3.client("ssm"))

        response = self._client.get_parameter(Name=key, WithDecryption=with_decryption)
        value = str(response["Parameter"]["Value"])
        self._cache[cache_key] = (now + self._cache_ttl, value)
        return value


_ssm_loader = _SSMParameterLoader()


def _get_laas_api_key() -> str:
    env_key = os.getenv("LAAS_API_KEY")
    if env_key:
        return env_key
    param_key = os.getenv("LAAS_API_KEY_SSM_PARAM", LAAS_API_KEY_SSM_PARAM)
    return _ssm_loader.get_parameter(param_key, True)


def _laas_post(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any] | list[Any]:
    base_url = os.getenv("LAAS_BASE_URL", LAAS_DEFAULT_BASE_URL).rstrip("/")
    url = f"{base_url}{path}"
    headers = {
        "project": "WANTED_DATA",
        "apiKey": _get_laas_api_key(),
        "Content-Type": "application/json; charset=utf-8",
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, (dict, list)):
            return data
        return {"content": data}


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
        response = _laas_post(f"/api/document/{collection}/similar/text", payload, timeout=30.0)
        return response if isinstance(response, list) else []
    except Exception as exc:
        logger.warning(
            "tools.rag_lookup.vector_search_failed",
            extra={"collection": collection, "error": str(exc)},
        )
        return []


def _join_doc_texts(docs: list[dict[str, Any]]) -> str:
    texts: list[str] = []
    for doc in docs:
        text = doc.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return "\n".join(texts)


def lookup_schema_rag_context(question: str) -> dict[str, Any]:
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
