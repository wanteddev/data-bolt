"""LangGraph node implementations for BigQuery agent."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, cast

from data_bolt.tasks import bigquery as bigquery_tasks
from data_bolt.tasks.relevance import should_respond_to_message

from .types import AgentState, ConversationMessage, Route, TurnAction

logger = logging.getLogger(__name__)

build_bigquery_sql = bigquery_tasks.build_bigquery_sql
plan_turn_action = bigquery_tasks.plan_turn_action
dry_run_bigquery_sql = bigquery_tasks.dry_run_bigquery_sql
execute_bigquery_sql = bigquery_tasks.execute_bigquery_sql
extract_sql_blocks = bigquery_tasks.extract_sql_blocks
plan_free_chat = bigquery_tasks.plan_free_chat
summarize_execution_result = bigquery_tasks.summarize_execution_result
explain_schema_lookup = bigquery_tasks.explain_schema_lookup
explain_sql_validation = bigquery_tasks.explain_sql_validation

_CONVERSATION_CLIP_LIMIT = int(os.getenv("BIGQUERY_AGENT_CONVERSATION_CLIP_LIMIT", "20"))
_BLOCKED_SQL_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE|REPLACE|GRANT|REVOKE|CALL|EXECUTE|BEGIN|COMMIT|ROLLBACK)\b",
    re.IGNORECASE,
)


def _env_truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _extract_user_sql(text: str) -> str | None:
    blocks = extract_sql_blocks(text, min_length=10)
    if blocks:
        sql = blocks[0].strip()
        return sql if sql.endswith(";") else f"{sql};"
    return None


def _clip_conversation(conversation: list[ConversationMessage]) -> list[ConversationMessage]:
    if _CONVERSATION_CLIP_LIMIT > 0 and len(conversation) > _CONVERSATION_CLIP_LIMIT:
        return conversation[-_CONVERSATION_CLIP_LIMIT:]
    return conversation


def _extract_sql_from_generation(result: dict[str, Any]) -> str | None:
    validation = result.get("validation") or {}
    validated_sql = validation.get("sql") if isinstance(validation, dict) else None
    if isinstance(validated_sql, str) and validated_sql.strip():
        return validated_sql.strip()

    structured = result.get("answer_structured") or {}
    sql = structured.get("sql")
    if isinstance(sql, str) and sql.strip():
        return sql.strip()
    return None


def _is_read_only_sql(sql: str) -> tuple[bool, str]:
    without_block_comments = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    normalized = re.sub(r"--[^\n]*", " ", without_block_comments).strip()
    statements = [part.strip() for part in normalized.split(";") if part.strip()]
    if not statements:
        return False, "실행할 SQL이 없습니다."
    if len(statements) != 1:
        return False, "다중 statement SQL은 실행할 수 없습니다."

    statement = statements[0]
    first_token = statement.split(None, 1)[0].upper() if statement.split() else ""
    if first_token not in {"SELECT", "WITH"}:
        return False, "읽기 전용 SELECT 쿼리만 실행할 수 있습니다."
    if _BLOCKED_SQL_KEYWORDS.search(statement):
        return False, "쓰기/DDL SQL은 실행할 수 없습니다."
    return True, ""


def _format_number(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "-"


def _coerce_str(value: Any, default: str = "") -> str:
    if isinstance(value, str):
        return value if value else default
    if value is None:
        return default
    normalized = str(value)
    return normalized if normalized else default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _env_truthy(value, default)
    if value is None:
        return default
    return bool(value)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _preview_cell(value: Any, max_len: int = 32) -> str:
    text = "-" if value is None else str(value)
    single_line = " ".join(text.split())
    if len(single_line) <= max_len:
        return single_line
    return f"{single_line[: max_len - 3]}..."


def _coerce_preview_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in value:
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _build_preview_table(
    rows: list[dict[str, Any]], max_rows: int = 8, max_cols: int = 6
) -> str | None:
    if not rows:
        return None
    first_row = rows[0]
    columns = [str(key) for key in first_row.keys()][:max_cols]
    if not columns:
        return None

    sample_rows = rows[:max_rows]
    cell_rows = [[_preview_cell(row.get(column)) for column in columns] for row in sample_rows]
    widths = []
    for index, column in enumerate(columns):
        column_width = len(column)
        for cells in cell_rows:
            column_width = max(column_width, len(cells[index]))
        widths.append(column_width)

    header = " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns))
    divider = "-+-".join("-" * widths[index] for index in range(len(columns)))
    body = [
        " | ".join(cells[index].ljust(widths[index]) for index in range(len(columns)))
        for cells in cell_rows
    ]
    return "\n".join([header, divider, *body])


def _result_insight_llm_enabled() -> bool:
    return _env_truthy(os.getenv("BIGQUERY_RESULT_INSIGHT_LLM_ENABLED"), True)


def _build_result_insight_sections(
    state: AgentState, preview_rows: list[dict[str, Any]]
) -> list[str]:
    if not preview_rows or not _result_insight_llm_enabled():
        return []

    execution = state.get("execution") or {}
    row_count = execution.get("row_count")
    sql = _coerce_str(state.get("candidate_sql")).strip()
    question = _coerce_str(state.get("text")).strip()
    history = cast(list[dict[str, Any]], state.get("conversation") or [])
    try:
        insight = summarize_execution_result(
            question=question,
            sql=sql,
            row_count=int(row_count) if isinstance(row_count, int | float) else None,
            preview_rows=preview_rows,
            history=history,
        )
    except Exception as exc:
        logger.warning("bigquery_agent.result_insight_fallback", extra={"error": str(exc)})
        return []

    sections: list[str] = []
    summary = insight.get("summary")
    if isinstance(summary, str) and summary.strip():
        sections.append(f":memo: 결과 요약\n- {summary.strip()}")

    insight_text = insight.get("insight")
    if isinstance(insight_text, str) and insight_text.strip():
        sections.append(f":mag: 데이터 해석\n- {insight_text.strip()}")

    follow_ups = insight.get("follow_up_questions")
    questions = [
        item.strip()
        for item in (follow_ups if isinstance(follow_ups, list) else [])
        if isinstance(item, str) and item.strip()
    ][:3]
    if questions:
        numbered = "\n".join(
            f"{index}. {question}" for index, question in enumerate(questions, start=1)
        )
        sections.append(f"다음 질문 제안\n{numbered}")

    return sections


def _get_auto_execute_max_cost_usd() -> float:
    raw = os.getenv("BIGQUERY_AUTO_EXECUTE_MAX_COST_USD", "1.0").strip()
    try:
        value = float(raw)
    except ValueError:
        return 1.0
    return value if value >= 0 else 1.0


def _normalize_history(value: Any) -> list[ConversationMessage]:
    if not isinstance(value, list):
        return []
    history: list[ConversationMessage] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if isinstance(role, str) and isinstance(content, str):
            history.append({"role": role, "content": content})
    return history


def _coerce_turn_action(value: Any, default: TurnAction = "chat_reply") -> TurnAction:
    if not isinstance(value, str):
        return default
    normalized = value.strip().lower()
    mapping: dict[str, TurnAction] = {
        "chat": "chat_reply",
        "free_chat": "chat_reply",
        "chat_reply": "chat_reply",
        "schema_lookup": "schema_lookup",
        "sql_validate_explain": "sql_validate_explain",
        "sql_generate": "sql_generate",
        "sql_execute": "sql_execute",
        "execution_approve": "execution_approve",
        "approve": "execution_approve",
        "execution_cancel": "execution_cancel",
        "cancel": "execution_cancel",
    }
    return mapping.get(normalized, default)


def _route_from_action(action: TurnAction) -> Route:
    if action == "chat_reply":
        return "chat"
    return "data"


def _build_routing_meta(state: AgentState) -> dict[str, Any]:
    return {
        "runtime_mode": state.get("runtime_mode") or "graph",
        "action": state.get("action") or "",
        "confidence": state.get("action_confidence", 0.0),
        "reason": state.get("action_reason") or "",
        "route": state.get("route") or "",
        "fallback_used": bool(state.get("fallback_used")),
        "prompt_version": "v2",
        "execution_policy": state.get("execution_policy") or "",
        "execution_policy_reason": state.get("execution_policy_reason") or "",
        "cost_threshold_usd": state.get("cost_threshold_usd"),
        "estimated_cost_usd": state.get("estimated_cost_usd"),
    }


def _node_reset_turn_state(state: AgentState) -> AgentState:
    return {
        "text": (state.get("text") or "").strip(),
        "should_respond": False,
        "action": "ignore",
        "user_sql": None,
        "candidate_sql": None,
        "dry_run": {},
        "generation_result": {},
        "can_execute": False,
        "execution": {},
        "response_text": "",
        "error": None,
        "route": "chat",
        "action_confidence": 0.0,
        "action_reason": "",
        "fallback_used": False,
        "execution_policy": "",
        "execution_policy_reason": "",
        "cost_threshold_usd": _get_auto_execute_max_cost_usd(),
        "estimated_cost_usd": None,
    }


def _node_clear_response(state: AgentState) -> AgentState:
    del state
    return {
        "should_respond": False,
        "response_text": "",
    }


def _node_ingest(state: AgentState) -> AgentState:
    text = (state.get("text") or "").strip()
    conversation = list(state.get("conversation") or [])
    history = state.get("history") or []

    if not conversation and history:
        conversation.extend(history)
    if text:
        conversation.append({"role": "user", "content": text})

    return {"text": text, "conversation": _clip_conversation(conversation)}


def _node_classify_relevance(state: AgentState) -> AgentState:
    should_respond = should_respond_to_message(
        text=state.get("text", ""),
        channel_type=state.get("channel_type", ""),
        is_mention=bool(state.get("is_mention")),
        is_thread_followup=bool(state.get("is_thread_followup")),
        channel_id=state.get("channel_id"),
    )
    if should_respond:
        return {"should_respond": True}
    return {"should_respond": False, "action": "ignore", "route": "chat"}


def _chat_planner_enabled() -> bool:
    return _env_truthy(os.getenv("BIGQUERY_CHAT_PLANNER_ENABLED"), True)


def _plan_chat_response(state: AgentState) -> str:
    text = _coerce_str(state.get("text")).strip()
    history = cast(list[dict[str, Any]], state.get("conversation") or [])
    allow_execute_in_chat = _env_truthy(os.getenv("BIGQUERY_CHAT_ALLOW_EXECUTE_IN_CHAT"), False)
    max_actions_raw = _coerce_str(os.getenv("BIGQUERY_CHAT_MAX_ACTIONS"), "2")
    try:
        max_actions = max(1, min(3, int(max_actions_raw)))
    except ValueError:
        max_actions = 2

    if _chat_planner_enabled():
        try:
            planned = plan_free_chat(
                text=text,
                history=history,
                allow_execute_in_chat=allow_execute_in_chat,
                max_actions=max_actions,
            )
            response = _coerce_str(planned.get("assistant_response")).strip()
            if response:
                return response
        except Exception as exc:
            logger.warning("bigquery_agent.chat_planner_fallback", extra={"error": str(exc)})

    return "질문 의도를 정확히 파악하지 못했습니다. 원하는 지표/기간/조건을 알려주시면 바로 도와드리겠습니다."


def _action_router_llm_enabled() -> bool:
    if "BIGQUERY_ACTION_ROUTER_LLM_ENABLED" in os.environ:
        return _env_truthy(os.getenv("BIGQUERY_ACTION_ROUTER_LLM_ENABLED"), True)
    return _env_truthy(os.getenv("BIGQUERY_INTENT_LLM_ENABLED"), True)


def _node_plan_turn_action(state: AgentState) -> AgentState:
    text = _coerce_str(state.get("text"))
    user_sql = _extract_user_sql(text)
    pending_execution_sql = _coerce_str(state.get("pending_execution_sql")).strip()

    if not _action_router_llm_enabled():
        return {
            "user_sql": user_sql,
            "action": "chat_reply",
            "route": "chat",
            "action_confidence": 0.0,
            "action_reason": "llm_router_disabled",
            "fallback_used": True,
        }

    try:
        raw = plan_turn_action(
            text=text,
            history=cast(list[dict[str, Any]], state.get("conversation") or []),
            channel_type=state.get("channel_type", ""),
            is_mention=bool(state.get("is_mention")),
            is_thread_followup=bool(state.get("is_thread_followup")),
            pending_execution_sql=pending_execution_sql or None,
            has_last_candidate_sql=bool(_coerce_str(state.get("last_candidate_sql")).strip()),
            has_last_dry_run=isinstance(state.get("last_dry_run"), dict)
            and "success" in cast(dict[str, Any], state.get("last_dry_run")),
            has_user_sql_block=bool(user_sql),
        )
        action = _coerce_turn_action(raw.get("action"), default="chat_reply")
        confidence_raw = raw.get("confidence")
        if confidence_raw is None:
            confidence = 0.0
        else:
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                confidence = 0.0
        reason = _coerce_str(raw.get("reason"), "llm routing")
        return {
            "user_sql": user_sql,
            "action": action,
            "route": _route_from_action(action),
            "action_confidence": confidence,
            "action_reason": reason,
            "fallback_used": False,
        }
    except Exception as exc:
        logger.warning("bigquery_agent.action_router_fallback", extra={"error": str(exc)})
        return {
            "user_sql": user_sql,
            "action": "chat_reply",
            "route": "chat",
            "action_confidence": 0.0,
            "action_reason": "router_error_safe_fallback",
            "fallback_used": True,
        }


def _node_chat_reply(state: AgentState) -> AgentState:
    response_text = _plan_chat_response(state)
    return {
        "route": "chat",
        "response_text": response_text,
        "generation_result": {
            "answer_structured": {
                "explanation": response_text,
                "instruction_type": "chat_reply",
            }
        },
    }


def _node_schema_lookup(state: AgentState) -> AgentState:
    text = _coerce_str(state.get("text"))
    table_info = _coerce_str(state.get("table_info"))
    glossary_info = _coerce_str(state.get("glossary_info"))
    history = cast(list[dict[str, Any]], state.get("conversation") or [])
    try:
        lookup = explain_schema_lookup(
            question=text,
            table_info=table_info,
            glossary_info=glossary_info,
            history=history,
        )
    except Exception as exc:
        logger.warning("bigquery_agent.schema_lookup_fallback", extra={"error": str(exc)})
        lookup = {
            "response_text": "스키마 정보를 불러오지 못했습니다. 조회하려는 지표/테이블을 구체적으로 알려주세요.",
            "reference_sql": None,
            "meta": {},
        }

    response_text = _coerce_str(lookup.get("response_text"), "").strip()
    reference_sql_raw = lookup.get("reference_sql")
    reference_sql = reference_sql_raw.strip() if isinstance(reference_sql_raw, str) else None
    meta_raw = lookup.get("meta")
    meta = meta_raw if isinstance(meta_raw, dict) else {}

    return {
        "route": "data",
        "candidate_sql": reference_sql,
        "generation_result": {
            "answer_structured": {
                "sql": reference_sql,
                "explanation": response_text,
                "instruction_type": "schema_lookup",
            },
            "meta": meta,
        },
        "response_text": response_text,
    }


def _node_sql_validate_explain(state: AgentState) -> AgentState:
    text = _coerce_str(state.get("text"))
    user_sql = state.get("user_sql")
    last_candidate_sql = _coerce_str(state.get("last_candidate_sql")).strip()
    last_dry_run_raw = state.get("last_dry_run")
    last_dry_run = last_dry_run_raw if isinstance(last_dry_run_raw, dict) else {}

    target_sql = _coerce_str(user_sql).strip() if isinstance(user_sql, str) else ""
    source = "user_sql"
    if not target_sql and last_candidate_sql:
        target_sql = last_candidate_sql
        source = "last_candidate_sql"

    if not target_sql:
        explanation = "검증/설명할 SQL이 없습니다. SQL 코드 블록을 보내거나 이전에 생성한 쿼리를 지정해주세요."
        return {
            "route": "data",
            "generation_result": {
                "answer_structured": {
                    "sql": None,
                    "explanation": explanation,
                    "instruction_type": "sql_validate_explain",
                }
            },
            "response_text": explanation,
        }

    if source == "user_sql":
        dry_run = dry_run_bigquery_sql(target_sql)
    elif last_dry_run and "success" in last_dry_run:
        dry_run = last_dry_run
    else:
        dry_run = dry_run_bigquery_sql(target_sql)

    if dry_run.get("success"):
        validated_sql = _coerce_str(dry_run.get("sql")).strip()
        if validated_sql:
            target_sql = validated_sql

    history = cast(list[dict[str, Any]], state.get("conversation") or [])
    try:
        explain_result = explain_sql_validation(
            question=text,
            sql=target_sql,
            dry_run=dry_run,
            history=history,
        )
    except Exception as exc:
        logger.warning("bigquery_agent.sql_validation_explain_fallback", extra={"error": str(exc)})
        explain_result = {"response_text": ""}

    explanation = _coerce_str(explain_result.get("response_text"), "").strip()
    if not explanation:
        if dry_run.get("success"):
            explanation = (
                "SQL dry-run 검증을 통과했습니다. 필요하면 비용/필터 조건을 더 점검해드릴게요."
            )
        else:
            explanation = f"SQL dry-run 검증에 실패했습니다. error: {dry_run.get('error', '-')}"

    result: AgentState = {
        "route": "data",
        "candidate_sql": target_sql,
        "dry_run": dry_run,
        "generation_result": {
            "answer_structured": {
                "sql": target_sql,
                "explanation": explanation,
                "instruction_type": "sql_validate_explain",
            }
        },
        "response_text": explanation,
    }
    if dry_run.get("success"):
        result["last_candidate_sql"] = target_sql
        result["last_dry_run"] = dry_run
    return result


def _node_sql_generate(state: AgentState) -> AgentState:
    text = _coerce_str(state.get("text"))
    payload: dict[str, Any] = {
        "text": text,
        "history": state.get("conversation") or [],
    }
    table_info = state.get("table_info")
    glossary_info = state.get("glossary_info")
    if table_info:
        payload["table_info"] = table_info
    if glossary_info:
        payload["glossary_info"] = glossary_info

    generation_result = build_bigquery_sql(payload)
    sql = _extract_sql_from_generation(generation_result)
    validation = generation_result.get("validation")
    dry_run_data: dict[str, Any] = validation if isinstance(validation, dict) else {}

    generated_candidate_sql: str | None = sql
    validation_sql = dry_run_data.get("sql")
    if isinstance(validation_sql, str) and validation_sql.strip():
        generated_candidate_sql = validation_sql.strip()

    output: AgentState = {
        "route": "data",
        "generation_result": generation_result,
        "candidate_sql": generated_candidate_sql,
        "dry_run": dry_run_data,
        "error": (
            generation_result.get("error")
            if isinstance(generation_result.get("error"), str)
            else None
        ),
    }
    if generated_candidate_sql and dry_run_data.get("success"):
        output["last_candidate_sql"] = generated_candidate_sql
        output["last_dry_run"] = dry_run_data
    return output


def _node_sql_execute(state: AgentState) -> AgentState:
    user_sql = state.get("user_sql")
    pending_execution_sql = _coerce_str(state.get("pending_execution_sql")).strip()
    pending_dry_run_raw = state.get("pending_execution_dry_run")
    pending_dry_run = pending_dry_run_raw if isinstance(pending_dry_run_raw, dict) else {}
    last_candidate_sql = _coerce_str(state.get("last_candidate_sql")).strip()
    last_dry_run_raw = state.get("last_dry_run")
    last_dry_run = last_dry_run_raw if isinstance(last_dry_run_raw, dict) else {}

    if isinstance(user_sql, str) and user_sql.strip():
        sql = user_sql.strip()
        dry_run = dry_run_bigquery_sql(sql)
        explanation = "요청에 포함된 SQL을 실행 준비를 위해 검증했습니다."
    elif pending_execution_sql:
        sql = pending_execution_sql
        dry_run = pending_dry_run if pending_dry_run else {}
        explanation = "승인 대기 중인 SQL을 실행 대상으로 불러왔습니다."
    elif last_candidate_sql:
        sql = last_candidate_sql
        dry_run = last_dry_run if last_dry_run else {}
        explanation = "이전 턴에서 생성한 SQL을 실행 대상으로 사용합니다."
    else:
        return {
            "route": "data",
            "candidate_sql": None,
            "execution": {
                "success": False,
                "error": "실행할 SQL이 없습니다. SQL 코드 블록을 제공하거나 이전 쿼리를 생성해주세요.",
            },
            "generation_result": {
                "answer_structured": {
                    "sql": None,
                    "explanation": "실행 요청을 처리할 SQL을 찾지 못했습니다.",
                    "instruction_type": "sql_execute",
                }
            },
        }

    if dry_run.get("success"):
        normalized_sql = _coerce_str(dry_run.get("sql")).strip()
        if normalized_sql:
            sql = normalized_sql

    result: AgentState = {
        "route": "data",
        "candidate_sql": sql,
        "dry_run": dry_run,
        "generation_result": {
            "answer_structured": {
                "sql": sql,
                "explanation": explanation,
                "instruction_type": "sql_execute",
            }
        },
    }
    if dry_run.get("success"):
        result["last_candidate_sql"] = sql
        result["last_dry_run"] = dry_run
    if not dry_run:
        result["response_text"] = (
            "실행 전 dry-run 정보가 없습니다. 먼저 SQL 검증을 수행하거나 쿼리를 다시 요청해주세요."
        )
    return result


def _node_validate_candidate_sql(state: AgentState) -> AgentState:
    sql = state.get("candidate_sql")
    if not isinstance(sql, str) or not sql.strip():
        return {}

    dry_run = state.get("dry_run")
    if isinstance(dry_run, dict) and "success" in dry_run:
        if dry_run.get("success"):
            normalized_sql = _coerce_str(dry_run.get("sql"), sql).strip() or sql
            return {"last_candidate_sql": normalized_sql, "last_dry_run": dry_run}
        return {}

    dry_run_result = dry_run_bigquery_sql(sql)
    if dry_run_result.get("success"):
        normalized_sql = _coerce_str(dry_run_result.get("sql"), sql).strip() or sql
        return {
            "dry_run": dry_run_result,
            "candidate_sql": normalized_sql,
            "last_candidate_sql": normalized_sql,
            "last_dry_run": dry_run_result,
        }
    return {"dry_run": dry_run_result}


def _node_policy_gate(state: AgentState) -> AgentState:
    action = state.get("action", "chat_reply")
    threshold = _get_auto_execute_max_cost_usd()

    approval_confirmed = action == "execution_approve"
    if action == "execution_cancel":
        return {
            "can_execute": False,
            "execution": {"success": False, "error": "쿼리 실행 요청을 취소했습니다."},
            "pending_execution_sql": None,
            "pending_execution_dry_run": {},
            "execution_policy": "blocked",
            "execution_policy_reason": "execution_cancelled",
            "cost_threshold_usd": threshold,
            "estimated_cost_usd": None,
        }

    if approval_confirmed:
        pending_sql = _coerce_str(state.get("pending_execution_sql")).strip()
        if not pending_sql:
            return {
                "can_execute": False,
                "execution": {
                    "success": False,
                    "error": "승인할 대기 중인 쿼리가 없습니다. 먼저 실행 요청을 보내주세요.",
                },
                "pending_execution_sql": None,
                "pending_execution_dry_run": {},
                "execution_policy": "blocked",
                "execution_policy_reason": "approval_without_pending_sql",
                "cost_threshold_usd": threshold,
                "estimated_cost_usd": None,
            }
        sql = pending_sql
        pending_dry_run_raw = state.get("pending_execution_dry_run")
        pending_dry_run = pending_dry_run_raw if isinstance(pending_dry_run_raw, dict) else {}
        dry_run_raw = state.get("dry_run")
        dry_run = dry_run_raw if isinstance(dry_run_raw, dict) else {}
        if not dry_run and pending_dry_run:
            dry_run = pending_dry_run
    elif action in {"sql_generate", "sql_execute"}:
        sql = _coerce_str(state.get("candidate_sql")).strip()
        dry_run_raw = state.get("dry_run")
        dry_run = dry_run_raw if isinstance(dry_run_raw, dict) else {}
    else:
        return {
            "can_execute": False,
            "execution_policy": "blocked",
            "execution_policy_reason": "action_not_executable",
            "cost_threshold_usd": threshold,
            "estimated_cost_usd": None,
        }

    if dry_run.get("success"):
        dry_run_sql = _coerce_str(dry_run.get("sql")).strip()
        if dry_run_sql:
            sql = dry_run_sql

    if not sql:
        execution = state.get("execution") or {}
        existing_error = execution.get("error")
        return {
            "can_execute": False,
            "execution": {
                "success": False,
                "error": existing_error or "실행할 SQL이 없습니다.",
            },
            "pending_execution_sql": None,
            "pending_execution_dry_run": {},
            "execution_policy": "blocked",
            "execution_policy_reason": "missing_sql",
            "cost_threshold_usd": threshold,
            "estimated_cost_usd": None,
        }

    read_only, reason = _is_read_only_sql(sql)
    if not read_only:
        return {
            "can_execute": False,
            "execution": {"success": False, "error": reason},
            "pending_execution_sql": None,
            "pending_execution_dry_run": {},
            "execution_policy": "blocked",
            "execution_policy_reason": "not_read_only",
            "cost_threshold_usd": threshold,
            "estimated_cost_usd": None,
        }

    if not dry_run:
        return {
            "can_execute": False,
            "execution": {"success": False, "error": "dry-run 정보가 없어 실행을 보류했습니다."},
            "pending_execution_sql": None,
            "pending_execution_dry_run": {},
            "execution_policy": "blocked",
            "execution_policy_reason": "missing_dry_run",
            "cost_threshold_usd": threshold,
            "estimated_cost_usd": None,
        }

    if not dry_run.get("success", True):
        return {
            "can_execute": False,
            "execution": {"success": False, "error": "dry-run 실패로 실행을 중단했습니다."},
            "pending_execution_sql": None,
            "pending_execution_dry_run": {},
            "execution_policy": "blocked",
            "execution_policy_reason": "dry_run_failed",
            "cost_threshold_usd": threshold,
            "estimated_cost_usd": _coerce_float(dry_run.get("estimated_cost_usd")),
        }

    max_bytes = int(os.getenv("BIGQUERY_MAX_BYTES_BILLED", "0"))
    bytes_processed = dry_run.get("total_bytes_processed")
    try:
        bytes_num = int(bytes_processed) if bytes_processed is not None else 0
    except (TypeError, ValueError):
        bytes_num = 0

    if max_bytes > 0 and bytes_num > max_bytes:
        return {
            "can_execute": False,
            "execution": {
                "success": False,
                "error": (
                    f"예상 처리 바이트가 제한을 초과했습니다. bytes={bytes_num}, limit={max_bytes}"
                ),
            },
            "pending_execution_sql": None,
            "pending_execution_dry_run": {},
            "execution_policy": "blocked",
            "execution_policy_reason": "max_bytes_exceeded",
            "cost_threshold_usd": threshold,
            "estimated_cost_usd": _coerce_float(dry_run.get("estimated_cost_usd")),
        }

    estimated_cost = _coerce_float(dry_run.get("estimated_cost_usd"))

    if approval_confirmed:
        return {
            "can_execute": True,
            "candidate_sql": sql,
            "dry_run": dry_run,
            "pending_execution_sql": None,
            "pending_execution_dry_run": {},
            "execution_policy": "auto_execute",
            "execution_policy_reason": "user_approved",
            "cost_threshold_usd": threshold,
            "estimated_cost_usd": estimated_cost,
        }

    if estimated_cost is not None and estimated_cost < threshold:
        return {
            "can_execute": True,
            "candidate_sql": sql,
            "dry_run": dry_run,
            "pending_execution_sql": None,
            "pending_execution_dry_run": {},
            "execution_policy": "auto_execute",
            "execution_policy_reason": "cost_below_threshold",
            "cost_threshold_usd": threshold,
            "estimated_cost_usd": estimated_cost,
        }

    if estimated_cost is None:
        approval_message = (
            "예상 비용을 계산할 수 없어 실행 전 사용자 승인이 필요합니다. "
            "위 쿼리와 dry-run 결과를 확인한 뒤 `실행 승인` 또는 `실행 취소`로 답변해주세요."
        )
        policy_reason = "estimated_cost_missing"
    else:
        approval_message = (
            "예상 비용이 자동 실행 임계값 이상입니다. "
            f"(estimated={estimated_cost}, threshold={threshold}) "
            "위 쿼리와 dry-run 결과를 확인한 뒤 `실행 승인` 또는 `실행 취소`로 답변해주세요."
        )
        policy_reason = "cost_above_threshold"

    return {
        "can_execute": False,
        "candidate_sql": sql,
        "dry_run": dry_run,
        "execution": {
            "success": False,
            "error": approval_message,
        },
        "pending_execution_sql": sql,
        "pending_execution_dry_run": dry_run,
        "execution_policy": "approval_required",
        "execution_policy_reason": policy_reason,
        "cost_threshold_usd": threshold,
        "estimated_cost_usd": estimated_cost,
    }


def _node_execute(state: AgentState) -> AgentState:
    if not state.get("can_execute"):
        return {}
    sql = state.get("candidate_sql")
    if not sql:
        return {"execution": {"success": False, "error": "실행할 SQL이 없습니다."}}
    return {
        "execution": execute_bigquery_sql(sql),
        "pending_execution_sql": None,
        "pending_execution_dry_run": {},
    }


def _build_sql_result_sections(state: AgentState) -> list[str]:
    sections: list[str] = []
    sql = state.get("candidate_sql")
    if isinstance(sql, str) and sql.strip():
        sections.append(f"```sql\n{sql.strip()}\n```")

    dry_run = state.get("dry_run") or {}
    if dry_run:
        if dry_run.get("success"):
            sections.append(
                ":white_check_mark: Dry-run 통과"
                f"\n- bytes processed: {_format_number(dry_run.get('total_bytes_processed'))}"
                f"\n- estimated cost (USD): {dry_run.get('estimated_cost_usd', '-')}"
            )
        else:
            sections.append(f":warning: Dry-run 실패\n- error: {dry_run.get('error', '-')}")

    execution = state.get("execution") or {}
    if execution:
        if execution.get("success"):
            preview_rows = _coerce_preview_rows(execution.get("preview_rows"))
            sections.append(
                ":rocket: 쿼리 실행 완료"
                f"\n- job_id: {execution.get('job_id', '-')}"
                f"\n- row_count: {_format_number(execution.get('row_count'))}"
            )
            if preview_rows:
                table = _build_preview_table(preview_rows)
                if table:
                    sections.append(f":bar_chart: 실행 결과 미리보기\n```text\n{table}\n```")
                sections.extend(_build_result_insight_sections(state, preview_rows))
            else:
                sections.append(":bar_chart: 실행 결과 미리보기가 비어 있습니다.")
        else:
            sections.append(
                f":warning: 쿼리 실행 생략/실패\n- reason: {execution.get('error', '-')}"
            )
    return sections


def _node_compose_response(state: AgentState) -> AgentState:
    if not state.get("should_respond"):
        return {"response_text": ""}

    result_raw = state.get("generation_result")
    result = result_raw if isinstance(result_raw, dict) else {}
    structured_raw = result.get("answer_structured")
    structured = structured_raw if isinstance(structured_raw, dict) else {}
    explanation_raw = structured.get("explanation")
    explanation = explanation_raw if isinstance(explanation_raw, str) else ""

    route = state.get("route", "data")
    if route == "chat":
        response_text = _coerce_str(state.get("response_text"), "")
        if not response_text:
            response_text = explanation.strip() or _plan_chat_response(state)
        conversation = list(state.get("conversation") or [])
        conversation.append({"role": "assistant", "content": response_text})
        return {"response_text": response_text, "conversation": _clip_conversation(conversation)}

    parts: list[str] = []
    error = state.get("error")
    if error:
        parts.append(f":x: SQL 생성 중 오류가 발생했습니다: {error}")
    elif explanation.strip():
        parts.append(explanation.strip())

    parts.extend(_build_sql_result_sections(state))

    if not parts:
        parts.append("요청을 처리할 수 있는 SQL 컨텍스트를 찾지 못했습니다.")
    response_text = "\n\n".join(parts)

    conversation = list(state.get("conversation") or [])
    conversation.append({"role": "assistant", "content": response_text})
    return {"response_text": response_text, "conversation": _clip_conversation(conversation)}


def _route_relevance(state: AgentState) -> str:
    return "proceed" if state.get("should_respond") else "stop"


def _route_action_node(state: AgentState) -> str:
    action = state.get("action", "chat_reply")
    mapping = {
        "chat_reply": "chat_reply",
        "schema_lookup": "schema_lookup",
        "sql_validate_explain": "sql_validate_explain",
        "sql_generate": "sql_generate",
        "sql_execute": "sql_execute",
        "execution_approve": "policy_gate",
        "execution_cancel": "policy_gate",
    }
    return mapping.get(action, "chat_reply")


def _route_execution(state: AgentState) -> str:
    if state.get("can_execute"):
        return "execute"
    return "skip"
