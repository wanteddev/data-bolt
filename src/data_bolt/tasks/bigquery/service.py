"""BigQuery SQL generation and validation service."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from . import execution, llm_client, parser, prompting, workflow_graph
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
from .tools import dry_run_tool, execute_query_tool, rag_context_tool
from .types import JsonValue, SQLBuildPayload, SQLWorkflowState

logger = logging.getLogger(__name__)

# Rebindable aliases for tests (monkeypatch compatibility).
_laas_post = llm_client._laas_post
_openai_compatible_post = llm_client._openai_compatible_post
_anthropic_compatible_post = llm_client._anthropic_compatible_post
_to_anthropic_messages = llm_client._to_anthropic_messages

_build_laas_messages = prompting._build_laas_messages
_classify_instruction_type = prompting._classify_instruction_type
_clip_history = prompting._clip_history
_build_refine_instruction_block = prompting._build_refine_instruction_block

_rag_context_tool = rag_context_tool
_dry_run_tool = dry_run_tool
_execute_query_tool = execute_query_tool


def _collect_rag_context(question: str) -> dict[str, Any]:
    return _rag_context_tool.run(question=question)


_extract_text_content = parser._extract_text_content
_collect_candidate_contents = parser._collect_candidate_contents
_parse_json_response = parser._parse_json_response
_extract_sql_from_response = parser._extract_sql_from_response

_ensure_trailing_semicolon = execution._ensure_trailing_semicolon


def dry_run_bigquery_sql(sql: str) -> dict[str, Any]:
    return _dry_run_tool.run(sql=sql)


def execute_bigquery_sql(sql: str) -> dict[str, Any]:
    return _execute_query_tool.run(sql=sql)


def _get_refine_attempts() -> int:
    raw = os.getenv("BIGQUERY_REFINE_MAX_ATTEMPTS", "3")
    try:
        value = int(raw)
    except ValueError:
        value = 3
    # Enforce bounded retries while allowing user-requested max retry loop.
    return max(0, min(3, value))


def _env_truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _use_sql_workflow_graph() -> bool:
    return _env_truthy(os.getenv("BIGQUERY_SQL_WORKFLOW_GRAPH_ENABLED"), False)


def _get_auto_execute_max_cost_usd() -> float:
    raw = os.getenv("BIGQUERY_AUTO_EXECUTE_MAX_COST_USD", "1.0").strip()
    try:
        value = float(raw)
    except ValueError:
        return 1.0
    return value if value >= 0 else 1.0


def _dry_run_sql(sql: str) -> tuple[bool, dict[str, Any]]:
    result = dry_run_bigquery_sql(sql)
    return bool(result.get("success")), {
        "error": result.get("error"),
        "total_bytes_processed": result.get("total_bytes_processed"),
        "job_id": result.get("job_id"),
        "cache_hit": result.get("cache_hit"),
    }


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


def plan_turn_action(
    *,
    text: str,
    history: list[dict[str, Any]] | None,
    channel_type: str,
    is_mention: bool,
    is_thread_followup: bool,
    pending_execution_sql: str | None = None,
    has_last_candidate_sql: bool = False,
    has_last_dry_run: bool = False,
    has_user_sql_block: bool = False,
) -> dict[str, Any]:
    clip_limit = int(os.getenv("BIGQUERY_HISTORY_CLIP_LIMIT", "6"))
    normalized_history = _clip_history(history, clip_limit)
    system_prompt = (
        """
너는 라우팅 전용 액션 플래너다. SQL 생성/실행은 절대 하지 말고 오직 JSON만 반환하라.

반드시 아래 스키마로 응답:
{
  "action": "chat_reply | schema_lookup | sql_validate_explain | sql_generate | sql_execute | execution_approve | execution_cancel",
  "intent_mode": "analysis | retrieval | execution | chat",
  "needs_clarification": false,
  "clarifying_question": "",
  "execution_intent": "none | suggested | explicit",
  "confidence": 0.0,
  "reason": "짧은 근거"
}

규칙:
- 일반 대화/아이데이션/메타 질문이면 action=chat_reply
- 분석/원인/해석 중심 요청이면 action=schema_lookup 또는 sql_validate_explain을 우선한다.
- 스키마/테이블/컬럼/조회 가능 항목 문의면 action=schema_lookup
- SQL 설명/검증/문법 점검 요청이면 action=sql_validate_explain
- 데이터 추출/집계 쿼리 생성 요청이라도 지표/서비스/기간이 모호하면 needs_clarification=true로 두고 action=chat_reply를 선택한다.
- 데이터 요청이 충분히 구체적일 때만 action=sql_generate
- 사용자 SQL 실행 요청이면 action=sql_execute, execution_intent=explicit
- 승인/취소 메시지이며 pending execution이 있으면 execution_approve 또는 execution_cancel
- sql_generate는 기본적으로 execution_intent=suggested로 둔다. 실행을 명시하지 않았다면 explicit로 올리지 않는다.
- 불확실하면 chat_reply를 선택하고 confidence를 낮게 설정
        """
    ).strip()
    user_prompt = (
        f"""
[Channel Meta]
- channel_type: {channel_type}
- is_mention: {is_mention}
- is_thread_followup: {is_thread_followup}

[Runtime State]
- pending_execution_sql_exists: {bool(pending_execution_sql)}
- has_last_candidate_sql: {has_last_candidate_sql}
- has_last_dry_run: {has_last_dry_run}
- has_user_sql_block: {has_user_sql_block}

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


def plan_free_chat(
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
너는 자유 대화 planner다. 자연스럽고 간결한 한국어 대화 응답을 반환하라.
반드시 JSON만 응답하고 스키마를 준수하라.

{
  "assistant_response": "자연스러운 한국어 답변"
}

규칙:
- 과도하게 장황하지 않게 답한다.
- 불확실한 사실은 단정하지 않는다.
- 다음 질문을 유도하고 싶다면 답변 문장 안에서 자연스럽게 제안한다.
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


def explain_schema_lookup(
    *,
    question: str,
    table_info: str,
    glossary_info: str,
    history: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    clip_limit = int(os.getenv("BIGQUERY_HISTORY_CLIP_LIMIT", "6"))
    normalized_history = _clip_history(history, clip_limit)

    rag_meta: dict[str, Any] = {"attempted": False}
    context_source = "provided"
    if question and (not table_info or not glossary_info):
        rag_ctx = _collect_rag_context(question=question)
        rag_meta_raw = rag_ctx.get("meta")
        rag_meta = rag_meta_raw if isinstance(rag_meta_raw, dict) else {"attempted": True}
        context_source = "rag"
        if not table_info:
            table_info = str(rag_ctx.get("table_info", ""))
        if not glossary_info:
            glossary_info = str(rag_ctx.get("glossary_info", ""))

    system_prompt = (
        """
너는 데이터 스키마 안내 도우미다. 실제 쿼리 실행은 절대 하지 말고 JSON만 반환하라.
반드시 아래 스키마를 따르라.

{
  "response_text": "현재 맥락에서 어떤 데이터 조회가 가능한지 설명",
  "reference_sql": "필요한 경우 참고용 read-only SQL, 없으면 빈 문자열"
}

규칙:
- 제공된 스키마/용어집 기준으로만 설명한다.
- 모르는 컬럼/테이블은 추정하지 않는다.
- reference_sql은 예시 목적의 단일 SELECT/CTE만 허용한다.
- 실행/비용/승인 안내가 필요하면 response_text에 짧게 포함한다.
        """
    ).strip()
    user_prompt = (
        f"""
[Question]
{question}

[Schema]
{table_info}

[Glossary]
{glossary_info}

[History]
{json.dumps(normalized_history, ensure_ascii=False)}
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
        return {
            "response_text": "",
            "reference_sql": None,
            "meta": {"context_source": context_source, "rag": rag_meta},
        }

    content = _extract_primary_text(resp)
    parsed = _parse_json_response(content)
    if not parsed:
        parsed = _parse_json_response(_extract_text_content(content) or "")

    response_text = ""
    reference_sql: str | None = None
    response_raw = parsed.get("response_text")
    if isinstance(response_raw, str):
        response_text = response_raw.strip()
    reference_raw = parsed.get("reference_sql")
    if isinstance(reference_raw, str) and reference_raw.strip():
        reference_sql = reference_raw.strip()

    return {
        "response_text": response_text,
        "reference_sql": reference_sql,
        "meta": {"context_source": context_source, "rag": rag_meta},
    }


def explain_sql_validation(
    *,
    question: str,
    sql: str,
    dry_run: dict[str, Any],
    history: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    clip_limit = int(os.getenv("BIGQUERY_HISTORY_CLIP_LIMIT", "6"))
    normalized_history = _clip_history(history, clip_limit)
    system_prompt = (
        """
너는 SQL 검증 결과를 설명하는 도우미다. 반드시 JSON만 응답하라.
출력 스키마:
{
  "response_text": "검증 결과 요약, 리스크, 다음 액션 제안"
}

규칙:
- dry_run 결과를 사실 기반으로 요약한다.
- 성공 시에도 잠재 리스크(비용/필터/집계)를 간단히 언급한다.
- 실패 시 원인과 수정 방향을 구체적으로 제시한다.
        """
    ).strip()
    user_prompt = (
        f"""
[Question]
{question}

[SQL]
{sql}

[DryRun]
{json.dumps(dry_run, ensure_ascii=False, default=str)}

[History]
{json.dumps(normalized_history, ensure_ascii=False)}
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
        return {"response_text": ""}

    content = _extract_primary_text(resp)
    parsed = _parse_json_response(content)
    if not parsed:
        parsed = _parse_json_response(_extract_text_content(content) or "")

    response_text = parsed.get("response_text")
    return {"response_text": response_text.strip() if isinstance(response_text, str) else ""}


def summarize_execution_result(
    *,
    question: str,
    sql: str,
    row_count: int | None,
    preview_rows: list[dict[str, Any]],
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    clip_limit = int(os.getenv("BIGQUERY_HISTORY_CLIP_LIMIT", "6"))
    normalized_history = _clip_history(history, clip_limit)
    preview_limit = int(os.getenv("BIGQUERY_RESULT_INSIGHT_PREVIEW_ROWS", "20"))
    sampled_preview = preview_rows[: max(1, preview_limit)]

    system_prompt = (
        """
너는 BigQuery 실행 결과를 설명하는 데이터 분석 어시스턴트다.
반드시 JSON만 응답하고 아래 스키마를 지켜라.

{
  "summary": "실행 결과 핵심 요약 1~2문장",
  "insight": "숫자 패턴/이상치/해석 포인트 1~2문장",
  "follow_up_questions": [
    "데이터를 더 잘 이해하기 위한 후속 질문 1",
    "후속 질문 2",
    "후속 질문 3"
  ]
}

규칙:
- 제공된 preview_rows와 row_count만 근거로 답한다. 모르는 값은 추정하지 않는다.
- row_count가 preview_rows 길이보다 크면 preview 기반 해석임을 문장에 반영한다.
- follow_up_questions는 데이터 내용에 맞는 구체적 질문으로 작성한다.
- SQL 수정 제안이 필요하면 질문 형태로 제시한다.
        """
    ).strip()
    user_prompt = (
        f"""
[Question]
{question}

[SQL]
{sql}

[row_count]
{row_count}

[preview_rows]
{json.dumps(sampled_preview, ensure_ascii=False, default=str)}

[History]
{json.dumps(normalized_history, ensure_ascii=False)}
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

    follow_ups_raw = parsed.get("follow_up_questions")
    follow_up_questions = [
        item.strip()
        for item in (follow_ups_raw if isinstance(follow_ups_raw, list) else [])
        if isinstance(item, str) and item.strip()
    ][:3]

    summary = parsed.get("summary")
    insight = parsed.get("insight")
    return {
        "summary": summary.strip() if isinstance(summary, str) else "",
        "insight": insight.strip() if isinstance(insight, str) else "",
        "follow_up_questions": follow_up_questions,
    }


def refine_bigquery_sql(
    *,
    question: str,
    ddl_context: str,
    prev_sql: str,
    error: str,
) -> str:
    system_prompt = _build_refine_instruction_block()
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
                        "error": None,
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


def _run_sql_generation_workflow(
    *,
    question: str,
    table_info: str,
    glossary_info: str,
    history: list[dict[str, Any]],
    images: list[dict[str, Any]],
    instruction_type: str,
    provider: str,
) -> SQLWorkflowState:
    deps = {
        "generate_bigquery_response": generate_bigquery_response,
        "adapt_response": adapt_llm_response_for_agent,
        "normalize_contract": _normalize_agent_response_contract,
        "dry_run_sql": _dry_run_sql,
        "refine_bigquery_sql": refine_bigquery_sql,
        "extract_sql_blocks": extract_sql_blocks,
        "ensure_trailing_semicolon": _ensure_trailing_semicolon,
        "estimate_query_cost_usd": estimate_query_cost_usd,
        "extract_llm_model": _extract_llm_model,
    }
    input_state: SQLWorkflowState = {
        "question": question,
        "table_info": table_info,
        "glossary_info": glossary_info,
        "history": history,
        "images": images,
        "instruction_type": instruction_type,
        "provider": provider,
        "max_refine_attempts": _get_refine_attempts(),
        "dependencies": deps,
    }
    return workflow_graph.run_sql_generation_workflow(input_state)


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
        rag_ctx = _collect_rag_context(question=question)
        rag_meta_raw = rag_ctx.get("meta")
        rag_meta = rag_meta_raw if isinstance(rag_meta_raw, dict) else {"attempted": True}
        context_source = "rag"
        if not table_info:
            table_info = rag_ctx.get("table_info", "")
        if not glossary_info:
            glossary_info = rag_ctx.get("glossary_info", "")
    workflow_enabled = _use_sql_workflow_graph()
    response: dict[str, Any]
    llm_meta: dict[str, Any]
    workflow_trace: list[str] = []
    if workflow_enabled:
        workflow_state = _run_sql_generation_workflow(
            question=question,
            table_info=table_info,
            glossary_info=glossary_info,
            history=history,
            images=images,
            instruction_type=instruction_type,
            provider=provider,
        )
        response = dict(workflow_state.get("response") or {})
        llm_meta_raw = workflow_state.get("llm_meta")
        llm_meta = llm_meta_raw if isinstance(llm_meta_raw, dict) else {"called": False}
        workflow_trace = list(workflow_state.get("workflow_trace") or [])
        validation = workflow_state.get("validation_meta")
        if (
            isinstance(validation, dict)
            and isinstance(workflow_state.get("generated_sql"), str)
            and workflow_state.get("generated_sql")
        ):
            response["validation"] = validation
    else:
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
                "meta": {},
            }
            response = _normalize_agent_response_contract(
                response, instruction_type=instruction_type, raw_resp=None
            )
        else:
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
            answer_structured = response.get("answer_structured")
            sql = answer_structured.get("sql") if isinstance(answer_structured, dict) else None

            if isinstance(sql, str) and sql:
                validation_result = _validate_with_refine(
                    question=question,
                    ddl_context=table_info,
                    glossary_info=glossary_info,
                    sql=sql,
                )
                response["validation"] = validation_result

    response["meta"] = {
        "instruction_type": instruction_type,
        "context_source": context_source,
        "rag": rag_meta,
        "llm": llm_meta,
        "laas": llm_meta if provider == LLM_PROVIDER_LAAS else {"called": False},
        "workflow": {
            "enabled": workflow_enabled,
            "trace": workflow_trace,
        },
    }

    return response
