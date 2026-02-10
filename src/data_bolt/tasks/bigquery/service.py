"""BigQuery SQL generation and validation service."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from . import execution, llm_client, parser, prompting, rag
from .execution import estimate_query_cost_usd
from .llm_config import (
    LAAS_EMPTY_PRESET_HASH,
    LLM_PROVIDER_LAAS,
    LLM_PROVIDER_OPENAI_COMPATIBLE,
    _get_anthropic_compatible_model,
    _get_llm_provider,
    _get_llm_timeout_seconds,
    _get_openai_compatible_model,
)
from .parser import extract_sql_blocks
from .types import JsonValue, SQLBuildPayload

logger = logging.getLogger(__name__)

# Rebindable aliases for tests (monkeypatch compatibility).
_laas_post = llm_client._laas_post
_openai_compatible_post = llm_client._openai_compatible_post
_anthropic_compatible_post = llm_client._anthropic_compatible_post
_to_anthropic_messages = llm_client._to_anthropic_messages

_build_laas_messages = prompting._build_laas_messages
_classify_instruction_type = prompting._classify_instruction_type
_clip_history = prompting._clip_history

_collect_rag_context = rag._collect_rag_context

_extract_text_content = parser._extract_text_content
_collect_candidate_contents = parser._collect_candidate_contents
_parse_json_response = parser._parse_json_response
_extract_sql_from_response = parser._extract_sql_from_response

_dry_run_sql = execution._dry_run_sql
_ensure_trailing_semicolon = execution._ensure_trailing_semicolon
dry_run_bigquery_sql = execution.dry_run_bigquery_sql
execute_bigquery_sql = execution.execute_bigquery_sql


def _get_refine_attempts() -> int:
    raw = os.getenv("BIGQUERY_REFINE_MAX_ATTEMPTS", "1")
    try:
        return max(0, int(raw))
    except ValueError:
        return 1


def _llm_chat_completion(
    *,
    messages: list[dict[str, Any]],
    timeout: float,
) -> JsonValue:
    provider = _get_llm_provider()
    if provider == "anthropic_compatible":
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


def _extract_primary_text(resp: dict[str, Any]) -> str:
    contents = _collect_candidate_contents(resp)
    return contents[0] if contents else ""


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


def adapt_llm_response_for_agent(raw_resp: JsonValue, instruction_type: str) -> dict[str, Any]:
    sql, explanation = _extract_sql_from_response(raw_resp)
    if not sql and not explanation and isinstance(raw_resp, dict):
        explanation = _extract_primary_text(raw_resp) or None
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


def _normalize_agent_response_contract(
    response: dict[str, Any], *, instruction_type: str, raw_resp: JsonValue | None
) -> dict[str, Any]:
    structured_raw = response.get("answer_structured")
    structured = structured_raw if isinstance(structured_raw, dict) else {}

    sql_raw = structured.get("sql")
    sql = sql_raw.strip() if isinstance(sql_raw, str) and sql_raw.strip() else None
    explanation_raw = structured.get("explanation")
    explanation = explanation_raw.strip() if isinstance(explanation_raw, str) else ""

    response["answer_structured"] = {
        "sql": sql,
        "explanation": explanation,
        "instruction_type": instruction_type,
    }
    response["choices"] = [{"message": {"content": sql or ""}}]
    response["raw_response"] = raw_resp if isinstance(raw_resp, (dict, list)) else {}
    return response


def _validate_with_refine(
    *,
    question: str,
    ddl_context: str,
    glossary_info: str,
    sql: str,
) -> dict[str, Any]:
    del glossary_info

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


def _extract_llm_model(raw_resp: JsonValue, provider: str) -> str:
    if not isinstance(raw_resp, dict):
        return ""
    model = raw_resp.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    if provider == LLM_PROVIDER_OPENAI_COMPATIBLE:
        return _get_openai_compatible_model()
    return ""


def build_bigquery_sql(payload: SQLBuildPayload | dict[str, Any]) -> dict[str, Any]:
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
        response = {
            "error": str(e),
            "meta": {
                "instruction_type": instruction_type,
                "context_source": context_source,
                "rag": rag_meta,
                "llm": llm_meta,
                "laas": llm_meta if provider == LLM_PROVIDER_LAAS else {"called": False},
            },
        }
        return _normalize_agent_response_contract(
            response, instruction_type=instruction_type, raw_resp=None
        )

    response = adapt_llm_response_for_agent(raw_resp, instruction_type)
    response = _normalize_agent_response_contract(
        response, instruction_type=instruction_type, raw_resp=raw_resp
    )
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
