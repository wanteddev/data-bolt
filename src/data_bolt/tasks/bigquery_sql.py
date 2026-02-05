"""BigQuery SQL generation and validation helpers."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

LAAS_DEFAULT_BASE_URL = "https://api-laas.wanted.co.kr"
LAAS_EMPTY_PRESET_HASH = (
    "2e1cfa82b035c26cbbbdae632cea070514eb8b773f616aaeaf668e2f0be8f10d"
)
LAAS_API_KEY_SSM_PARAM = "/DATA/PIPELINE/API_KEY/OPENAI"
LAAS_RAG_SCHEMA_COLLECTION = "RAG_DATA_CATALOG"
LAAS_RAG_GLOSSARY_COLLECTION = "RAG_GLOSSARY"


def extract_sql_blocks(text: str, min_length: int = 20) -> list[str]:
    """Extract SQL blocks from a text blob."""
    if not text:
        return []

    blocks: list[str] = []

    fenced_pattern = re.compile(r"```(?:sql)?\n(.*?)```", re.S | re.I)
    for match in fenced_pattern.finditer(text):
        block = match.group(1).strip()
        if len(block) >= min_length:
            blocks.extend(
                [b.strip() for b in block.split(";") if len(b.strip()) >= min_length]
            )

    if blocks:
        return blocks

    sql_start_pattern = re.compile(r"^(SELECT|WITH|CREATE|INSERT|UPDATE|DELETE)\b", re.I)
    parts = [p.strip() for p in re.split(r";", text) if p.strip()]
    for part in parts:
        first_line = part.strip().splitlines()[0].strip() if part.strip() else ""
        if sql_start_pattern.search(first_line):
            blocks.append(part)

    return blocks


def _ensure_trailing_semicolon(sql: Optional[str]) -> Optional[str]:
    if sql is None:
        return None
    s = sql.rstrip()
    return s if s.endswith(";") else s + ";"


def _looks_like_sql(text: str) -> bool:
    if not text:
        return False
    if "```" in text:
        return True
    pat = re.compile(r"\b(SELECT|WITH|CREATE|INSERT|UPDATE|DELETE)\b", re.I)
    return bool(pat.search(text))


def _classify_instruction_type(question: str, history: list[dict[str, Any]] | None) -> str:
    q = (question or "").lower()
    wants_analysis = any(
        k in q
        for k in [
            "분석",
            "해석",
            "설명",
            "왜",
            "느려",
            "에러",
            "error",
            "explain",
            "optimize",
        ]
    )
    has_sql = _looks_like_sql(question) or any(
        _looks_like_sql(m.get("content", ""))
        for m in (history or [])
        if m.get("role") == "user"
    )
    if wants_analysis and has_sql:
        return "bigquery_sql_analysis"
    return "bigquery_sql_generation"


def _env_truthy(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_refine_attempts() -> int:
    raw = os.getenv("BIGQUERY_REFINE_MAX_ATTEMPTS", "1")
    try:
        return max(0, int(raw))
    except ValueError:
        return 1


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {"content": resp.text}


def _build_instruction_block(instruction_type: str) -> str:
    if instruction_type == "bigquery_sql_generation":
        return (
            """
[Directive]
너는 BigQuery 표준 SQL을 작성하는 데이터 엔지니어 도우미다.
반드시 아래 규칙과 절차를 따른다.

[응답 규칙]
1) BigQuery 표준 SQL만 사용 (legacy 금지), SELECT * 금지, 스키마에 없는 컬럼 추측 금지.
2) 한국 시간(Asia/Seoul) 고려 시 TIMESTAMP/DATETIME 변환을 명시.
3) 조인 시 키와 null 처리 근거를 설명에 적시.
4) 결과는 단일 쿼리 또는 CTE로 제공. 임시 테이블 금지.
5) 출력 JSON 스키마:
{
  "sql": "<BigQuery SQL>",
  "explanation": "<요청 해석, 조인/필터 근거, 시간대 처리 근거>",
  "assumptions": "<제공되지 않은 가정 목록 또는 빈 배열/문자열>",
  "validation_steps": [
    "스키마 존재 확인 방법",
    "작은 기간으로 샘플 실행해 행수/NULL 비율 검증",
    "엣지 케이스 점검 아이디어"
  ]
}
6) COUNT(_) 금지, COUNT(*) 사용. 7) 필요한 컬럼만 SELECT.

[절차]
1) 요청 재진술 → 2) 컨텍스트 선택 → 3) 설계 설명 → 4) SQL 생성 → 5) 자체 검증.
            """
        ).strip()
    if instruction_type == "bigquery_sql_analysis":
        return (
            """
[Directive]
당신은 데이터 분석 전문가로써 요청에 맞게 응답을 제공한다.
답변은 언제나 읽기 쉽고 이해하기 쉬워야 하며 질문자의 의도에 부합해야 한다.
최대한 질문자를 도와주기 위해서 최선을 다해야 합니다.
당신은 차가운 기계가 아닌 질문자의 최선을 다해서 도우려는 따뜻한 동료입니다.
결과는 반드시 아래 JSON 스키마를 따를 것.

동일한 출력 JSON 스키마:
{
  "sql": "<개선된 BigQuery SQL 또는 새 SQL 또는 빈 문자열>",
  "explanation": "<문제 진단 및 개선 근거>",
  "assumptions": "<가정 목록>",
  "validation_steps": [
    "스키마/키 유효성 점검",
    "작은 기간 샘플 실행",
    "엣지 케이스 검토"
  ]
}
성능/정확성 지침은 생성 지시문과 동일하게 따른다.
            """
        ).strip()
    return (
        """
[Directive]
아래 JSON 스키마를 따르는 BigQuery 표준 SQL을 생성하라.
{
  "sql": "<BigQuery SQL>",
  "explanation": "<요청 해석 및 근거>",
  "assumptions": "<가정>",
  "validation_steps": ["체크리스트"]
}
        """
    ).strip()


def _clip_history(
    history: list[dict[str, Any]] | None, clip_limit: int
) -> list[dict[str, Any]]:
    if not history:
        return []
    normalized: list[dict[str, Any]] = []
    for message in history:
        try:
            role = message.get("role")
            if role in ("user", "assistant"):
                normalized.append({"role": role, "content": message.get("content")})
        except Exception:
            continue
    if clip_limit > 0 and len(normalized) > clip_limit:
        return normalized[-clip_limit:]
    return normalized


def _build_laas_messages(
    *,
    question: str,
    table_info: str,
    glossary_info: str,
    history: list[dict[str, Any]] | None,
    images: list[dict[str, Any]] | None,
    instruction_type: str,
) -> list[dict[str, Any]]:
    context_block = (
        f"""
[Context]
DDL / Schema:
{table_info}

Glossary / Business Rules:
{glossary_info}
        """
    ).strip()
    instruction_block = _build_instruction_block(instruction_type)
    clip_limit = int(os.getenv("BIGQUERY_HISTORY_CLIP_LIMIT", "6"))
    normalized_history = _clip_history(history, clip_limit)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": context_block},
        {"role": "system", "content": instruction_block},
        *normalized_history,
        {"role": "user", "content": (question or "").strip()},
    ]
    if images:
        messages.append({"role": "user", "content": list(images)})
    return messages


class _SSMParameterLoader:
    def __init__(self, cache_ttl: int = 300):
        self._cache: dict[str, tuple[float, str]] = {}
        self._cache_ttl = cache_ttl
        self._client = None

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

            self._client = boto3.client("ssm")

        response = self._client.get_parameter(
            Name=key, WithDecryption=with_decryption
        )
        value = response["Parameter"]["Value"]
        self._cache[cache_key] = (now + self._cache_ttl, value)
        return value


_ssm_loader = _SSMParameterLoader()


def _get_laas_api_key() -> str:
    env_key = os.getenv("LAAS_API_KEY")
    if env_key:
        return env_key
    param_key = os.getenv("LAAS_API_KEY_SSM_PARAM", LAAS_API_KEY_SSM_PARAM)
    return _ssm_loader.get_parameter(param_key, True)


def _laas_post(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    base_url = os.getenv("LAAS_BASE_URL", LAAS_DEFAULT_BASE_URL).rstrip("/")
    api_key = _get_laas_api_key()

    url = f"{base_url}{path}"
    headers = {
        "project": "WANTED_DATA",
        "apiKey": api_key,
        "Content-Type": "application/json; charset=utf-8",
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


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
        resp = _laas_post(
            f"/api/document/{collection}/similar/text", payload, timeout=30.0
        )
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


def _collect_rag_context(question: str) -> dict[str, str]:
    schema_collection = os.getenv(
        "LAAS_RAG_SCHEMA_COLLECTION", LAAS_RAG_SCHEMA_COLLECTION
    )
    glossary_collection = os.getenv(
        "LAAS_RAG_GLOSSARY_COLLECTION", LAAS_RAG_GLOSSARY_COLLECTION
    )
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

    return {
        "table_info": _join_doc_texts(schema_docs),
        "glossary_info": _join_doc_texts(glossary_docs),
    }


def generate_bigquery_response(
    *,
    question: str,
    table_info: str,
    glossary_info: str,
    history: list[dict[str, Any]] | None,
    images: list[dict[str, Any]] | None,
    instruction_type: str,
) -> dict[str, Any]:
    messages = _build_laas_messages(
        question=question,
        table_info=table_info,
        glossary_info=glossary_info,
        history=history,
        images=images,
        instruction_type=instruction_type,
    )
    payload = {"hash": LAAS_EMPTY_PRESET_HASH, "messages": messages}
    return _laas_post("/api/preset/v2/chat/completions", payload, timeout=60.0)


def refine_bigquery_sql(
    *,
    question: str,
    ddl_context: str,
    prev_sql: str,
    error: str,
) -> str:
    system_prompt = (
        """
너는 **BigQuery 표준 SQL**을 작성하는 데이터 엔지니어 도우미다. 아래 “응답 규칙”과 “절차”를 항상 지켜라.

[응답 규칙]
1) BigQuery 표준 SQL만 사용 (legacy 금지), SELECT * 금지, 스키마에 없는 컬럼 추측 금지.
2) 한국 시간(Asia/Seoul) 기준 시간 해석이 필요하면 TIMESTAMP/DATETIME 변환을 명시.
3) 조인 시 키와 null 처리 근거를 설명에 적시.
4) 결과는 반드시 단일 쿼리 또는 CTE로 제공. 임시 테이블 금지.
5) 출력 형식은 아래 JSON 스키마를 따를 것:
{
  "sql": "<BigQuery SQL>",
  "explanation": "<요청 해석, 조인/필터 근거, 시간대 처리 근거>",
  "assumptions": "<제공되지 않은 가정이 있다면 나열, 없으면 빈 배열 또는 빈 문자열>",
  "validation_steps": [
    "스키마 존재 확인 방법",
    "작은 기간으로 샘플 실행해 행수/NULL 비율 검증",
    "엣지 케이스 점검 아이디어"
  ]
}
6) BigQuery standard SQL 사용, legacy SQL 금지
7) COUNT(_) 사용 금지, COUNT(*) 사용
8) 타임존 변환 명시 (DATETIME(TIMESTAMP(...), "Asia/Seoul"))
9) 성능 고려: 조인 시 필요한 컬럼만 SELECT

[절차]
1) 요청 분석: 비즈니스 질문을 한 문장으로 재진술.
2) 컨텍스트 선택: 제공된 DDL/용어집에서 필요한 부분만 인용.
3) 설계: 조인 키, 필터, 집계, 타임존 처리 방식을 글머리표로 설명.
4) 생성: 설계대로 SQL 작성.
5) 자체 검증: 체크리스트로 점검 후 필요 시 수정.
        """
    ).strip()
    user_prompt = (
        f"""
### 1) 비즈니스 요청
{question}

### 2) DDL / 스키마
{ddl_context}

### 3) 이전 SQL
{prev_sql if prev_sql else ""}

### 4) 에러 메시지
{error if error else ""}
        """
    ).strip()
    payload = {
        "hash": LAAS_EMPTY_PRESET_HASH,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    try:
        resp = _laas_post("/api/preset/v2/chat/completions", payload, timeout=60.0)
        content = (
            (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
            if isinstance(resp, dict)
            else ""
        )
        parsed = _parse_json_response(content)
        sql = parsed.get("sql")
        return sql if isinstance(sql, str) else ""
    except Exception:
        return ""


def _dry_run_sql(sql: str) -> tuple[bool, dict[str, Any]]:
    url = os.getenv("BIGQUERY_DRYRUN_URL")
    if not url:
        return False, {"error": "BIGQUERY_DRYRUN_URL is not set"}

    payload = {"sql": sql, "dry_run": True}
    try:
        resp = _post_json(url, payload, timeout=30.0)
        ok = resp.get("success")
        if ok is None:
            ok = resp.get("ok")
        if ok is None:
            ok = "error" not in resp
        return bool(ok), {
            "total_bytes_processed": resp.get("total_bytes_processed"),
            "job_id": resp.get("job_id"),
            "cache_hit": resp.get("cache_hit"),
            "error": resp.get("error"),
        }
    except Exception as e:
        return False, {"error": str(e)}


def _extract_sql_from_response(resp: dict[str, Any]) -> tuple[str | None, str | None]:
    sql = resp.get("sql") if isinstance(resp, dict) else None
    explanation = resp.get("explanation") if isinstance(resp, dict) else None
    if isinstance(sql, str) and sql.strip():
        return sql.strip(), explanation

    choices = (resp.get("choices") or []) if isinstance(resp, dict) else []
    if choices:
        message = (choices[0] or {}).get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parsed = _parse_json_response(content)
            parsed_sql = parsed.get("sql")
            parsed_explanation = parsed.get("explanation")
            if isinstance(parsed_sql, str) and parsed_sql.strip():
                return parsed_sql.strip(), parsed_explanation
            blocks = extract_sql_blocks(content)
            return (blocks[0] if blocks else content.strip()), explanation

    return None, explanation


def _parse_json_response(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except Exception:
        return {}


def _validation_enabled() -> bool:
    env = os.getenv("ENABLE_BIGQUERY_DRYRUN_VALIDATION")
    if not _env_truthy(env, True):
        return False
    return bool(os.getenv("BIGQUERY_DRYRUN_URL"))


def _validate_with_refine(
    *,
    question: str,
    ddl_context: str,
    glossary_info: str,
    sql: str,
) -> dict[str, Any]:
    validation_meta: dict[str, Any] = {
        "success": False,
        "refined": False,
        "attempts": 0,
        "sql": None,
        "error": None,
        "total_bytes_processed": None,
        "cache_hit": False,
        "job_id": None,
        "refinement_error": None,
    }

    validation_meta["attempts"] += 1
    ok, meta = _dry_run_sql(sql)
    if ok:
        validation_meta.update(
            {
                "success": True,
                "sql": _ensure_trailing_semicolon(sql),
                "total_bytes_processed": meta.get("total_bytes_processed"),
                "job_id": meta.get("job_id"),
                "cache_hit": meta.get("cache_hit"),
            }
        )
        return validation_meta

    validation_meta["error"] = meta.get("error")

    max_refine = _get_refine_attempts()
    refine_attempted = 0
    while refine_attempted < max_refine:
        refine_attempted += 1
        validation_meta["attempts"] += 1
        refined_text = refine_bigquery_sql(
            question=question,
            ddl_context=ddl_context,
            prev_sql=sql,
            error=validation_meta.get("error") or "",
        )
        refined_blocks = extract_sql_blocks(refined_text or "")
        for candidate in refined_blocks:
            ok, meta = _dry_run_sql(candidate)
            if ok:
                validation_meta.update(
                    {
                        "success": True,
                        "refined": True,
                        "sql": _ensure_trailing_semicolon(candidate),
                        "total_bytes_processed": meta.get("total_bytes_processed"),
                        "job_id": meta.get("job_id"),
                        "cache_hit": meta.get("cache_hit"),
                    }
                )
                return validation_meta
            validation_meta["error"] = meta.get("error")
        if not refined_blocks:
            validation_meta["refinement_error"] = "Refine produced no SQL blocks"

    return validation_meta


def build_bigquery_sql(payload: dict[str, Any]) -> dict[str, Any]:
    question = payload.get("text") or payload.get("question") or ""
    table_info = payload.get("table_info", "")
    glossary_info = payload.get("glossary_info", "")
    history = payload.get("history") or []
    images = payload.get("images") or []

    if question and (not table_info or not glossary_info):
        rag_ctx = _collect_rag_context(question)
        if not table_info:
            table_info = rag_ctx.get("table_info", "")
        if not glossary_info:
            glossary_info = rag_ctx.get("glossary_info", "")

    instruction_type = _classify_instruction_type(question, history)
    try:
        raw_resp = generate_bigquery_response(
            question=question,
            table_info=table_info,
            glossary_info=glossary_info,
            history=history,
            images=images,
            instruction_type=instruction_type,
        )
    except Exception as e:
        logger.error("bigquery_sql.generate_failed", extra={"error": str(e)})
        return {"error": str(e)}

    sql, explanation = _extract_sql_from_response(raw_resp)
    response: dict[str, Any] = {
        "choices": [{"message": {"content": (sql or "").strip()}}],
        "answer_structured": {
            "sql": sql,
            "explanation": explanation,
            "instruction_type": instruction_type,
        },
        "raw_response": raw_resp,
    }

    if sql and _validation_enabled():
        validation = _validate_with_refine(
            question=question,
            ddl_context=table_info,
            glossary_info=glossary_info,
            sql=sql,
        )
        response["validation"] = validation

    return response
