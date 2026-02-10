"""BigQuery SQL generation and validation helpers."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any

import httpx
from google.api_core.exceptions import BadRequest, GoogleAPICallError

if TYPE_CHECKING:
    from google.cloud.bigquery import Client as BigQueryClient
    from mypy_boto3_ssm import SSMClient

JsonValue = dict[str, Any] | list[Any]

logger = logging.getLogger(__name__)

LAAS_DEFAULT_BASE_URL = "https://api-laas.wanted.co.kr"
LAAS_EMPTY_PRESET_HASH = "2e1cfa82b035c26cbbbdae632cea070514eb8b773f616aaeaf668e2f0be8f10d"
LAAS_API_KEY_SSM_PARAM = "/DATA/PIPELINE/API_KEY/OPENAI"
LAAS_RAG_SCHEMA_COLLECTION = "RAG_DATA_CATALOG"
LAAS_RAG_GLOSSARY_COLLECTION = "RAG_GLOSSARY"
LLM_PROVIDER_DEFAULT = "laas"
LLM_PROVIDER_LAAS = "laas"
LLM_PROVIDER_OPENAI_COMPATIBLE = "openai_compatible"
LLM_PROVIDER_ANTHROPIC_COMPATIBLE = "anthropic_compatible"
OPENAI_COMPATIBLE_DEFAULT_MODEL = "glm-4.7"
ANTHROPIC_COMPATIBLE_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
LLM_TIMEOUT_SECONDS_DEFAULT = 60.0
LLM_TIMEOUT_INTENT_SECONDS_DEFAULT = 45.0
LLM_TIMEOUT_CHAT_PLANNER_SECONDS_DEFAULT = 45.0
LLM_TIMEOUT_REFINE_SECONDS_DEFAULT = 60.0
BIGQUERY_ON_DEMAND_USD_PER_TB = float(os.getenv("BIGQUERY_ON_DEMAND_USD_PER_TB", "5"))


def extract_sql_blocks(text: str, min_length: int = 20) -> list[str]:
    """Extract SQL blocks from a text blob."""
    if not text:
        return []

    blocks: list[str] = []

    fenced_pattern = re.compile(r"```(?:sql)?\n(.*?)```", re.S | re.I)
    for match in fenced_pattern.finditer(text):
        block = match.group(1).strip()
        if len(block) >= min_length:
            blocks.extend([b.strip() for b in block.split(";") if len(b.strip()) >= min_length])

    if blocks:
        return blocks

    sql_start_pattern = re.compile(r"^(SELECT|WITH|CREATE|INSERT|UPDATE|DELETE)\b", re.I)
    parts = [p.strip() for p in re.split(r";", text) if p.strip()]
    for part in parts:
        first_line = part.strip().splitlines()[0].strip() if part.strip() else ""
        if sql_start_pattern.search(first_line):
            blocks.append(part)

    return blocks


def _ensure_trailing_semicolon(sql: str | None) -> str | None:
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
        _looks_like_sql(m.get("content", "")) for m in (history or []) if m.get("role") == "user"
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


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def estimate_query_cost_usd(
    total_bytes_processed: int | str | None,
    *,
    price_per_tb_usd: float = BIGQUERY_ON_DEMAND_USD_PER_TB,
) -> float | None:
    bytes_processed = _coerce_int(total_bytes_processed)
    if bytes_processed is None or bytes_processed < 0:
        return None
    tebibyte = float(1024**4)
    return round((bytes_processed / tebibyte) * price_per_tb_usd, 6)


def _build_instruction_block(instruction_type: str) -> str:
    if instruction_type == "general_chat":
        return (
            """
[Directive]
너는 Slack에서 대화를 돕는 친절한 데이터 동료다.
일상 대화에는 자연스럽게 답하고, 데이터/쿼리 요청이 나오면 필요한 조건을 짧게 확인해라.
사실을 추정해 단정하지 말고, 모르면 모른다고 말하라.

출력은 반드시 아래 JSON 스키마를 따를 것:
{
  "sql": "",
  "explanation": "<자연스러운 한국어 대화 응답>",
  "assumptions": [],
  "validation_steps": []
}
            """
        ).strip()
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


def _clip_history(history: list[dict[str, Any]] | None, clip_limit: int) -> list[dict[str, Any]]:
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

            self._client = boto3.client("ssm")

        response = self._client.get_parameter(Name=key, WithDecryption=with_decryption)
        value = str(response["Parameter"]["Value"])
        self._cache[cache_key] = (now + self._cache_ttl, value)
        return value


_ssm_loader = _SSMParameterLoader()
_bigquery_client: BigQueryClient | None = None


def _get_laas_api_key() -> str:
    env_key = os.getenv("LAAS_API_KEY")
    if env_key:
        return env_key
    param_key = os.getenv("LAAS_API_KEY_SSM_PARAM", LAAS_API_KEY_SSM_PARAM)
    return _ssm_loader.get_parameter(param_key, True)


def _get_llm_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER_DEFAULT).strip().lower()
    if provider in {
        LLM_PROVIDER_LAAS,
        LLM_PROVIDER_OPENAI_COMPATIBLE,
        LLM_PROVIDER_ANTHROPIC_COMPATIBLE,
    }:
        return provider
    return LLM_PROVIDER_DEFAULT


def _get_openai_compatible_api_key() -> str:
    value = os.getenv("LLM_OPENAI_API_KEY", "").strip()
    if value:
        return value
    raise ValueError("LLM_OPENAI_API_KEY is required when LLM_PROVIDER=openai_compatible")


def _get_openai_compatible_base_url() -> str:
    value = os.getenv("LLM_OPENAI_BASE_URL", "").strip().rstrip("/")
    if value:
        return value
    raise ValueError("LLM_OPENAI_BASE_URL is required when LLM_PROVIDER=openai_compatible")


def _get_openai_compatible_model() -> str:
    model = os.getenv("LLM_OPENAI_MODEL", OPENAI_COMPATIBLE_DEFAULT_MODEL).strip()
    return model or OPENAI_COMPATIBLE_DEFAULT_MODEL


def _get_anthropic_compatible_api_key() -> str:
    value = os.getenv("LLM_ANTHROPIC_API_KEY", "").strip()
    if value:
        return value
    raise ValueError("LLM_ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic_compatible")


def _get_anthropic_compatible_base_url() -> str:
    value = os.getenv("LLM_ANTHROPIC_BASE_URL", "").strip().rstrip("/")
    if value:
        return value
    raise ValueError("LLM_ANTHROPIC_BASE_URL is required when LLM_PROVIDER=anthropic_compatible")


def _get_anthropic_compatible_model() -> str:
    model = os.getenv("LLM_ANTHROPIC_MODEL", ANTHROPIC_COMPATIBLE_DEFAULT_MODEL).strip()
    return model or ANTHROPIC_COMPATIBLE_DEFAULT_MODEL


def _read_timeout_env(var_name: str, default: float) -> float:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _read_timeout_env_optional(var_name: str) -> float | None:
    raw = os.getenv(var_name)
    if raw is None:
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _get_llm_timeout_seconds(use_case: str) -> float:
    common = _read_timeout_env_optional("LLM_TIMEOUT_SECONDS")
    if use_case == "intent":
        specific = _read_timeout_env_optional("LLM_TIMEOUT_INTENT_SECONDS")
        return specific if specific is not None else common or LLM_TIMEOUT_INTENT_SECONDS_DEFAULT
    if use_case == "planner":
        specific = _read_timeout_env_optional("LLM_TIMEOUT_CHAT_PLANNER_SECONDS")
        return (
            specific if specific is not None else common or LLM_TIMEOUT_CHAT_PLANNER_SECONDS_DEFAULT
        )
    if use_case == "refine":
        specific = _read_timeout_env_optional("LLM_TIMEOUT_REFINE_SECONDS")
        return specific if specific is not None else common or LLM_TIMEOUT_REFINE_SECONDS_DEFAULT
    specific = _read_timeout_env_optional("LLM_TIMEOUT_GENERATION_SECONDS")
    return specific if specific is not None else common or LLM_TIMEOUT_SECONDS_DEFAULT


def _get_bigquery_client() -> BigQueryClient:
    global _bigquery_client
    if _bigquery_client is None:
        from google.cloud import bigquery

        project = os.getenv("BIGQUERY_PROJECT_ID") or None
        location = os.getenv("BIGQUERY_LOCATION") or None
        if location:
            _bigquery_client = bigquery.Client(project=project, location=location)
        else:
            _bigquery_client = bigquery.Client(project=project)
    return _bigquery_client


def _get_bigquery_location() -> str | None:
    value = os.getenv("BIGQUERY_LOCATION")
    if value and value.strip():
        return value.strip()
    return None


def _get_query_timeout_seconds() -> float:
    raw = os.getenv("BIGQUERY_QUERY_TIMEOUT_SECONDS", "120")
    try:
        timeout = float(raw)
    except ValueError:
        timeout = 120.0
    return timeout if timeout > 0 else 120.0


def _get_max_bytes_billed() -> int | None:
    raw = os.getenv("BIGQUERY_MAX_BYTES_BILLED", "0").strip()
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _format_bigquery_error(error: Exception) -> str:
    if isinstance(error, BadRequest):
        errors = getattr(error, "errors", None)
        if isinstance(errors, list) and errors:
            first = errors[0] if isinstance(errors[0], dict) else {}
            message = first.get("message")
            if isinstance(message, str) and message.strip():
                return message
        if str(error).strip():
            return str(error)
        return "BigQuery dry-run failed with bad request."
    if isinstance(error, GoogleAPICallError) and str(error).strip():
        return str(error)
    return str(error) if str(error).strip() else "Unknown BigQuery error"


def _laas_post(path: str, payload: dict[str, Any], timeout: float) -> JsonValue:
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
        data = resp.json()
        if isinstance(data, (dict, list)):
            return data
        return {"content": data}


def _openai_compatible_post(path: str, payload: dict[str, Any], timeout: float) -> JsonValue:
    base_url = _get_openai_compatible_base_url()
    api_key = _get_openai_compatible_api_key()

    url = f"{base_url}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json; charset=utf-8",
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, (dict, list)):
            return data
        return {"content": data}


def _anthropic_compatible_post(path: str, payload: dict[str, Any], timeout: float) -> JsonValue:
    base_url = _get_anthropic_compatible_base_url()
    api_key = _get_anthropic_compatible_api_key()

    url = f"{base_url}{path}"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json; charset=utf-8",
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, (dict, list)):
            return data
        return {"content": data}


def _to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    out_messages: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = _extract_text_content(content)
            if text:
                system_parts.append(text)
            continue
        if role not in {"user", "assistant"}:
            continue
        text = _extract_text_content(content)
        if not text:
            continue
        out_messages.append({"role": role, "content": text})
    return ("\n\n".join(system_parts) if system_parts else None), out_messages


def _llm_chat_completion(
    *,
    messages: list[dict[str, Any]],
    timeout: float,
) -> JsonValue:
    provider = _get_llm_provider()
    if provider == LLM_PROVIDER_ANTHROPIC_COMPATIBLE:
        system, anthropic_messages = _to_anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": _get_anthropic_compatible_model(),
            "max_tokens": int(os.getenv("LLM_ANTHROPIC_MAX_TOKENS", "4096")),
            "messages": anthropic_messages or [{"role": "user", "content": ""}],
        }
        if system:
            payload["system"] = system
        return _anthropic_compatible_post("/v1/messages", payload, timeout=timeout)
    if provider == LLM_PROVIDER_OPENAI_COMPATIBLE:
        payload = {
            "model": _get_openai_compatible_model(),
            "messages": messages,
        }
        return _openai_compatible_post("/chat/completions", payload, timeout=timeout)

    payload = {"hash": LAAS_EMPTY_PRESET_HASH, "messages": messages}
    return _laas_post("/api/preset/v2/chat/completions", payload, timeout=timeout)


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
    resp = _llm_chat_completion(
        messages=messages,
        timeout=_get_llm_timeout_seconds("generation"),
    )
    return resp if isinstance(resp, dict) else {"choices": []}


def classify_intent_with_laas(
    *,
    text: str,
    history: list[dict[str, Any]] | None,
    channel_type: str,
    is_mention: bool,
    is_thread_followup: bool,
) -> dict[str, Any]:
    clip_limit = int(os.getenv("BIGQUERY_HISTORY_CLIP_LIMIT", "6"))
    normalized_history = _clip_history(history, clip_limit)
    system_prompt = (
        """
너는 라우팅 전용 분류기다. SQL 생성/실행은 절대 하지 말고 오직 JSON만 반환하라.

반드시 아래 스키마로 응답:
{
  "intent": "data_workflow | free_chat",
  "confidence": 0.0,
  "reason": "짧은 근거",
  "actions": ["text_to_sql" | "validate_sql" | "execute_sql" | "schema_lookup" | "analysis_followup" | "none"]
}

규칙:
- 데이터 조회/집계/스키마/쿼리 생성·검증·실행 요청이면 intent=data_workflow
- 일반 대화/메타 질문이면 intent=free_chat
- 불확실하면 intent=free_chat, confidence를 낮게
- actions는 intent에 맞게 1~2개, 해당 없으면 ["none"]
        """
    ).strip()
    user_prompt = (
        f"""
[Channel Meta]
- channel_type: {channel_type}
- is_mention: {is_mention}
- is_thread_followup: {is_thread_followup}

[History]
{json.dumps(normalized_history, ensure_ascii=False)}

[User]
{text}
        """
    ).strip()
    resp = _llm_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        timeout=_get_llm_timeout_seconds("intent"),
    )
    if not isinstance(resp, dict):
        return {}
    content = _extract_primary_text(resp)
    parsed = _parse_json_response(content)
    if not parsed:
        parsed = _parse_json_response(_extract_text_content(content) or "")
    return parsed


def plan_free_chat_with_laas(
    *,
    text: str,
    history: list[dict[str, Any]] | None,
    allow_execute_in_chat: bool,
    max_actions: int,
) -> dict[str, Any]:
    clip_limit = int(os.getenv("BIGQUERY_HISTORY_CLIP_LIMIT", "6"))
    normalized_history = _clip_history(history, clip_limit)
    system_prompt = (
        """
너는 자유 대화 planner다. 대화 응답을 만들고 필요할 때만 액션을 제안하라.
반드시 JSON만 응답하고 스키마를 준수하라.

{
  "assistant_response": "자연스러운 한국어 답변",
  "actions": ["text_to_sql" | "validate_sql" | "execute_sql" | "schema_lookup" | "analysis_followup" | "none"],
  "action_reason": "액션 선택 근거"
}

규칙:
- 액션은 최대 지정 개수만 반환.
- 실행 액션은 정책상 허용될 때만 추천.
- 데이터 작업이 필요 없으면 actions=["none"].
        """
    ).strip()
    user_prompt = (
        f"""
[Policy]
- allow_execute_in_chat: {allow_execute_in_chat}
- max_actions: {max_actions}

[History]
{json.dumps(normalized_history, ensure_ascii=False)}

[User]
{text}
        """
    ).strip()
    resp = _llm_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        timeout=_get_llm_timeout_seconds("planner"),
    )
    if not isinstance(resp, dict):
        return {}
    content = _extract_primary_text(resp)
    parsed = _parse_json_response(content)
    if not parsed:
        parsed = _parse_json_response(_extract_text_content(content) or "")
    return parsed


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
    try:
        resp = _llm_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=_get_llm_timeout_seconds("refine"),
        )
        content = _extract_primary_text(resp) if isinstance(resp, dict) else ""
        parsed = _parse_json_response(content)
        sql = parsed.get("sql")
        return sql if isinstance(sql, str) else ""
    except Exception:
        return ""


def _dry_run_sql(sql: str) -> tuple[bool, dict[str, Any]]:
    if not sql or not sql.strip():
        return False, {"error": "SQL is empty"}

    try:
        from google.cloud import bigquery

        client = _get_bigquery_client()
        config = bigquery.QueryJobConfig(
            dry_run=True,
            use_query_cache=False,
            use_legacy_sql=False,
        )
        max_bytes = _get_max_bytes_billed()
        if max_bytes is not None:
            config.maximum_bytes_billed = max_bytes
        query_job = client.query(
            sql,
            job_config=config,
            location=_get_bigquery_location(),
        )
        if query_job.errors:
            first_error = query_job.errors[0] if query_job.errors else {}
            error_text = (
                str(first_error.get("message"))
                if isinstance(first_error, dict) and first_error.get("message")
                else "BigQuery dry-run returned errors"
            )
            return False, {"error": error_text}
        return True, {
            "total_bytes_processed": query_job.total_bytes_processed,
            "job_id": query_job.job_id,
            "cache_hit": query_job.cache_hit,
            "error": None,
        }
    except Exception as e:
        return False, {"error": _format_bigquery_error(e)}


def dry_run_bigquery_sql(sql: str) -> dict[str, Any]:
    ok, meta = _dry_run_sql(sql)
    return {
        "success": ok,
        "sql": _ensure_trailing_semicolon(sql) if ok else None,
        "error": meta.get("error"),
        "total_bytes_processed": meta.get("total_bytes_processed"),
        "estimated_cost_usd": estimate_query_cost_usd(meta.get("total_bytes_processed")),
        "job_id": meta.get("job_id"),
        "cache_hit": meta.get("cache_hit"),
    }


def execute_bigquery_sql(sql: str) -> dict[str, Any]:
    try:
        from google.cloud import bigquery

        client = _get_bigquery_client()
        config = bigquery.QueryJobConfig(use_legacy_sql=False)
        max_bytes = _get_max_bytes_billed()
        if max_bytes is not None:
            config.maximum_bytes_billed = max_bytes
        query_job = client.query(
            sql,
            job_config=config,
            location=_get_bigquery_location(),
        )
        iterator = query_job.result(timeout=_get_query_timeout_seconds())
        preview_limit = int(os.getenv("BIGQUERY_RESULT_PREVIEW_ROWS", "20"))
        preview_rows: list[dict[str, Any]] = []
        for index, row in enumerate(iterator):
            if index < preview_limit:
                preview_rows.append(dict(row.items()))
            else:
                break

        row_count = iterator.total_rows
        if row_count is None:
            row_count = query_job.num_dml_affected_rows

        return {
            "success": True,
            "error": None,
            "job_id": query_job.job_id,
            "row_count": row_count,
            "preview_rows": preview_rows,
        }
    except Exception as e:
        return {
            "success": False,
            "error": _format_bigquery_error(e),
            "job_id": None,
            "row_count": None,
            "preview_rows": [],
        }


def _extract_sql_from_response(resp: JsonValue) -> tuple[str | None, str | None]:
    sql = resp.get("sql") if isinstance(resp, dict) else None
    explanation = resp.get("explanation") if isinstance(resp, dict) else None
    if isinstance(sql, str) and sql.strip():
        return sql.strip(), explanation

    contents = _collect_candidate_contents(resp)
    for content in contents:
        parsed = _parse_json_response(content)
        if parsed and "sql" in parsed:
            parsed_sql = parsed.get("sql")
            parsed_explanation = parsed.get("explanation")
            resolved_explanation = (
                parsed_explanation
                if isinstance(parsed_explanation, str)
                else explanation
                if isinstance(explanation, str)
                else None
            )
            if isinstance(parsed_sql, str):
                normalized_sql = parsed_sql.strip()
                return (normalized_sql or None), resolved_explanation
            return None, resolved_explanation

        parsed_sql = parsed.get("sql")
        parsed_explanation = parsed.get("explanation")
        if isinstance(parsed_sql, str) and parsed_sql.strip():
            resolved_explanation = (
                parsed_explanation
                if isinstance(parsed_explanation, str)
                else explanation
                if isinstance(explanation, str)
                else None
            )
            return parsed_sql.strip(), resolved_explanation

        blocks = extract_sql_blocks(content)
        if blocks:
            return blocks[0], explanation
        if content.strip():
            return content.strip(), explanation

    return None, explanation


def _extract_text_content(content: Any) -> str | None:
    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _extract_text_content(item)
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)
        return None

    if isinstance(content, dict):
        for key in ("text", "content", "value", "output_text"):
            text = _extract_text_content(content.get(key))
            if text:
                return text
    return None


def _collect_candidate_contents(resp: JsonValue) -> list[str]:
    if not isinstance(resp, dict):
        return []

    candidates: list[str] = []

    for key in ("content", "text", "output_text"):
        text = _extract_text_content(resp.get(key))
        if text:
            candidates.append(text)

    message_text = _extract_text_content(resp.get("message"))
    if message_text:
        candidates.append(message_text)

    choices = resp.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                text = _extract_text_content(message.get("content"))
                if text:
                    candidates.append(text)
            text = _extract_text_content(choice.get("content"))
            if text:
                candidates.append(text)

    output = resp.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            text = _extract_text_content(item.get("content"))
            if text:
                candidates.append(text)

    return candidates


def _extract_primary_text(resp: dict[str, Any]) -> str:
    contents = _collect_candidate_contents(resp)
    return contents[0] if contents else ""


def _extract_llm_model(raw_resp: JsonValue, provider: str) -> str:
    if not isinstance(raw_resp, dict):
        return ""
    model = raw_resp.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    if provider == LLM_PROVIDER_OPENAI_COMPATIBLE:
        return _get_openai_compatible_model()
    return ""


def adapt_llm_response_for_agent(raw_resp: JsonValue, instruction_type: str) -> dict[str, Any]:
    sql, explanation = _extract_sql_from_response(raw_resp)
    return {
        "choices": [{"message": {"content": (sql or "").strip()}}],
        "answer_structured": {
            "sql": sql,
            "explanation": explanation,
            "instruction_type": instruction_type,
        },
        "raw_response": raw_resp,
    }


def adapt_laas_response_for_agent(raw_resp: JsonValue, instruction_type: str) -> dict[str, Any]:
    return adapt_llm_response_for_agent(raw_resp, instruction_type)


def _parse_json_response(content: str) -> dict[str, Any]:
    candidates = [content]
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.S | re.I)
    if fenced_match:
        candidates.append(fenced_match.group(1))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return {}


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
        "estimated_cost_usd": None,
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
                "estimated_cost_usd": estimate_query_cost_usd(meta.get("total_bytes_processed")),
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
                        "estimated_cost_usd": estimate_query_cost_usd(
                            meta.get("total_bytes_processed")
                        ),
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
    raw_instruction_type = payload.get("instruction_type")
    instruction_type = (
        raw_instruction_type
        if isinstance(raw_instruction_type, str) and raw_instruction_type.strip()
        else _classify_instruction_type(question, history)
    )
    rag_meta: dict[str, Any] = {"attempted": False}
    context_source = "provided"
    provider = _get_llm_provider()

    should_collect_rag = instruction_type != "general_chat"
    if should_collect_rag and question and (not table_info or not glossary_info):
        rag_ctx = _collect_rag_context(question)
        rag_meta_raw = rag_ctx.get("meta")
        rag_meta = rag_meta_raw if isinstance(rag_meta_raw, dict) else {"attempted": True}
        context_source = "rag"
        if not table_info:
            table_info = rag_ctx.get("table_info", "")
        if not glossary_info:
            glossary_info = rag_ctx.get("glossary_info", "")
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
        llm_meta = {"provider": provider, "called": True, "success": False, "error": str(e)}
        return {
            "error": str(e),
            "meta": {
                "instruction_type": instruction_type,
                "context_source": context_source,
                "rag": rag_meta,
                "llm": llm_meta,
                "laas": llm_meta if provider == LLM_PROVIDER_LAAS else {"called": False},
            },
        }

    response = adapt_llm_response_for_agent(raw_resp, instruction_type)
    llm_meta = {
        "provider": provider,
        "called": True,
        "success": True,
        "model": _extract_llm_model(raw_resp, provider),
    }
    response["meta"] = {
        "instruction_type": instruction_type,
        "context_source": context_source,
        "rag": rag_meta,
        "llm": llm_meta,
        "laas": llm_meta if provider == LLM_PROVIDER_LAAS else {"called": False},
    }
    answer_structured = response.get("answer_structured")
    sql = answer_structured.get("sql") if isinstance(answer_structured, dict) else None

    if isinstance(sql, str) and sql:
        validation = _validate_with_refine(
            question=question,
            ddl_context=table_info,
            glossary_info=glossary_info,
            sql=sql,
        )
        response["validation"] = validation

    return response
