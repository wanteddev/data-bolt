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


def _coerce_analysis_brief(value: Any) -> dict[str, Any]:
    brief = value if isinstance(value, dict) else {}
    goal = _coerce_str(brief.get("goal")).strip()
    metric_definition = _coerce_str(brief.get("metric_definition")).strip()
    scope = _coerce_str(brief.get("scope")).strip()
    time_window = _coerce_str(brief.get("time_window")).strip()
    latest_findings = _coerce_str_list(brief.get("latest_findings"), limit=5)
    open_questions = _coerce_str_list(brief.get("open_questions"), limit=5)
    next_actions = _coerce_str_list(brief.get("next_recommended_actions"), limit=3)
    return {
        "goal": goal,
        "metric_definition": metric_definition,
        "scope": scope,
        "time_window": time_window,
        "latest_findings": latest_findings,
        "open_questions": open_questions,
        "next_recommended_actions": next_actions,
    }


def _coerce_str_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    items = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return items[:limit]


def _infer_intent_mode(action: TurnAction) -> str:
    if action in {"schema_lookup", "sql_validate_explain"}:
        return "analysis"
    if action == "sql_generate":
        return "retrieval"
    if action in {"sql_execute", "execution_approve", "execution_cancel"}:
        return "execution"
    return "chat"


def _coerce_intent_mode(value: Any, *, action: TurnAction) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"analysis", "retrieval", "execution", "chat"}:
            return normalized
    return _infer_intent_mode(action)


def _coerce_execution_intent(value: Any, *, text: str, action: TurnAction) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"none", "suggested", "explicit"}:
            return normalized
    if action in {"sql_execute", "execution_approve", "execution_cancel"}:
        return "explicit"
    explicit_signal = bool(
        re.search(r"(실행(해|해줘|해주세요|해 볼|해봐)|돌려(줘|주세요)|run|execute)", text, re.I)
    )
    if explicit_signal:
        return "explicit"
    if action == "sql_generate":
        return "suggested"
    return "none"


def _needs_clarification_fallback(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    has_analysis_signal = any(
        token in normalized for token in ("분석", "해석", "원인", "인사이트", "추이", "비교")
    )
    has_scope_signal = any(
        token in normalized for token in ("서비스", "채널", "세그먼트", "플랫폼")
    )
    has_time_signal = any(
        token in normalized
        for token in ("어제", "오늘", "지난", "최근", "주", "월", "년", "기간", "date")
    )
    has_metric_signal = any(
        token in normalized for token in ("가입", "사용자", "매출", "전환", "건수", "비율", "지표")
    )
    # 분석 의도는 있는데 지표/범위/기간이 충분히 명시되지 않은 경우.
    return has_analysis_signal and not ((has_scope_signal and has_time_signal) or has_metric_signal)


def _extract_llm_follow_up_suggestions(result: dict[str, Any]) -> list[str]:
    structured_raw = result.get("answer_structured")
    structured = structured_raw if isinstance(structured_raw, dict) else {}
    candidates = structured.get("follow_up_questions")
    if not isinstance(candidates, list):
        candidates = structured.get("next_recommended_actions")
    parsed = _coerce_str_list(candidates, limit=3)
    deduped: list[str] = []
    for item in parsed:
        if item not in deduped:
            deduped.append(item)
    return deduped[:3]


def _update_analysis_brief(state: AgentState, follow_ups: list[str]) -> dict[str, Any]:
    brief = _coerce_analysis_brief(state.get("analysis_brief"))
    text = _coerce_str(state.get("text")).strip()
    if text and not brief.get("goal"):
        brief["goal"] = text

    execution = state.get("execution") if isinstance(state.get("execution"), dict) else {}
    dry_run = state.get("dry_run") if isinstance(state.get("dry_run"), dict) else {}
    findings = list(brief.get("latest_findings", []))
    open_questions = list(brief.get("open_questions", []))
    if dry_run:
        if dry_run.get("success"):
            findings.append(
                "dry-run 통과: "
                f"bytes={_format_number(dry_run.get('total_bytes_processed'))}, "
                f"cost={dry_run.get('estimated_cost_usd', '-')}"
            )
        else:
            open_questions.append(f"dry-run 실패 원인 확인 필요: {dry_run.get('error', '-')}")
    if execution:
        if execution.get("success"):
            findings.append(
                "쿼리 실행 완료: "
                f"row_count={_format_number(execution.get('row_count'))}, "
                f"job_id={execution.get('job_id', '-')}"
            )
        else:
            error = _coerce_str(execution.get("error")).strip()
            if error:
                open_questions.append(error)
    clarifying_question = _coerce_str(state.get("clarifying_question")).strip()
    if clarifying_question:
        open_questions.append(clarifying_question)
    brief["latest_findings"] = findings[-5:]
    brief["open_questions"] = open_questions[-5:]
    brief["next_recommended_actions"] = follow_ups[:3]
    return brief


def _coerce_turn_action(value: Any, default: TurnAction = "chat_reply") -> TurnAction:
    if not isinstance(value, str):
        return default
    normalized = value.strip().lower()
    mapping: dict[str, TurnAction] = {
        "ignore": "ignore",
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


def _mode_from_action(action: TurnAction) -> str:
    if action == "chat_reply":
        return "chat"
    if action in {"schema_lookup", "sql_validate_explain"}:
        return "analyze"
    if action in {"sql_generate", "sql_execute", "execution_approve", "execution_cancel"}:
        return "execute"
    return "chat"


def _planned_tool_from_action(action: TurnAction) -> str:
    mapping: dict[TurnAction, str] = {
        "ignore": "",
        "chat_reply": "",
        "schema_lookup": "schema_lookup",
        "sql_validate_explain": "sql_validate_explain",
        "sql_generate": "sql_execute",
        "sql_execute": "sql_execute",
        "execution_approve": "execution_approve",
        "execution_cancel": "execution_cancel",
    }
    return mapping.get(action, "")


def _planned_tool_for_decision(action: TurnAction, execution_intent: str) -> str:
    if action == "sql_generate" and execution_intent != "explicit":
        return ""
    return _planned_tool_from_action(action)


def _build_routing_meta(state: AgentState) -> dict[str, Any]:
    action = _coerce_turn_action(state.get("action"), default="ignore")
    execution_intent = _coerce_str(state.get("execution_intent"), "none").strip().lower() or "none"
    return {
        "runtime_mode": state.get("runtime_mode") or "loop",
        "action": action,
        "intent_mode": state.get("intent_mode") or _infer_intent_mode(action),
        "execution_intent": execution_intent,
        "needs_clarification": bool(state.get("needs_clarification")),
        "turn_mode": state.get("turn_mode") or _mode_from_action(action),
        "planned_tool": state.get("planned_tool")
        or _planned_tool_for_decision(action, execution_intent),
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
        "intent_mode": "chat",
        "execution_intent": "none",
        "needs_clarification": False,
        "clarifying_question": "",
        "action": "ignore",
        "user_sql": None,
        "candidate_sql": None,
        "dry_run": {},
        "generation_result": {},
        "execution": {},
        "response_text": "",
        "error": None,
        "route": "chat",
        "action_confidence": 0.0,
        "action_reason": "",
        "fallback_used": False,
        "turn_mode": "chat",
        "planned_tool": "",
        "execution_policy": "",
        "execution_policy_reason": "",
        "cost_threshold_usd": _get_auto_execute_max_cost_usd(),
        "estimated_cost_usd": None,
        "analysis_brief": _coerce_analysis_brief(state.get("analysis_brief")),
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


def _node_decide_turn(
    state: AgentState, *, check_relevance: bool = True, plan_action: bool = True
) -> AgentState:
    text = _coerce_str(state.get("text"))
    user_sql = _extract_user_sql(text)
    should_respond = True
    if check_relevance:
        should_respond = should_respond_to_message(
            text=text,
            channel_type=state.get("channel_type", ""),
            is_mention=bool(state.get("is_mention")),
            is_thread_followup=bool(state.get("is_thread_followup")),
            channel_id=state.get("channel_id"),
        )
    if not should_respond:
        return {
            "should_respond": False,
            "intent_mode": "chat",
            "execution_intent": "none",
            "needs_clarification": False,
            "clarifying_question": "",
            "action": "ignore",
            "route": "chat",
            "turn_mode": "chat",
            "planned_tool": "",
        }
    if not plan_action:
        return {"should_respond": True}

    pending_execution_sql = _coerce_str(state.get("pending_execution_sql")).strip()
    if not _action_router_llm_enabled():
        action: TurnAction = "chat_reply"
        intent_mode = _infer_intent_mode(action)
        execution_intent = _coerce_execution_intent(
            None,
            text=text,
            action=action,
        )
        return {
            "should_respond": True,
            "intent_mode": intent_mode,
            "execution_intent": execution_intent,
            "needs_clarification": False,
            "clarifying_question": "",
            "user_sql": user_sql,
            "action": action,
            "route": _route_from_action(action),
            "action_confidence": 0.0,
            "action_reason": "llm_router_disabled",
            "fallback_used": True,
            "turn_mode": _mode_from_action(action),
            "planned_tool": _planned_tool_for_decision(action, execution_intent),
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
        intent_mode = _coerce_intent_mode(raw.get("intent_mode"), action=action)
        needs_clarification = bool(raw.get("needs_clarification"))
        clarifying_question = _coerce_str(raw.get("clarifying_question")).strip()
        execution_intent = _coerce_execution_intent(
            raw.get("execution_intent"),
            text=text,
            action=action,
        )
        if (
            action == "sql_generate"
            and execution_intent != "explicit"
            and not needs_clarification
            and _needs_clarification_fallback(text)
        ):
            needs_clarification = True
        if needs_clarification and not clarifying_question:
            clarifying_question = (
                "좋아요. 분석 방향을 정확히 맞추기 위해 기준을 먼저 정할게요. "
                "어떤 서비스/지표/기간을 우선으로 볼까요?"
            )
        if needs_clarification and clarifying_question:
            action = "chat_reply"
            intent_mode = "analysis"
            execution_intent = "none"
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
            "should_respond": True,
            "intent_mode": intent_mode,
            "execution_intent": execution_intent,
            "needs_clarification": needs_clarification and bool(clarifying_question),
            "clarifying_question": clarifying_question,
            "user_sql": user_sql,
            "action": action,
            "route": _route_from_action(action),
            "action_confidence": confidence,
            "action_reason": reason,
            "fallback_used": False,
            "turn_mode": _mode_from_action(action),
            "planned_tool": _planned_tool_for_decision(action, execution_intent),
        }
    except Exception as exc:
        logger.warning("bigquery_agent.action_router_fallback", extra={"error": str(exc)})
        action = "chat_reply"
        intent_mode = _infer_intent_mode(action)
        execution_intent = _coerce_execution_intent(
            None,
            text=text,
            action=action,
        )
        return {
            "should_respond": True,
            "intent_mode": intent_mode,
            "execution_intent": execution_intent,
            "needs_clarification": False,
            "clarifying_question": "",
            "user_sql": user_sql,
            "action": action,
            "route": _route_from_action(action),
            "action_confidence": 0.0,
            "action_reason": "router_error_safe_fallback",
            "fallback_used": True,
            "turn_mode": _mode_from_action(action),
            "planned_tool": _planned_tool_for_decision(action, execution_intent),
        }


def _node_chat_reply(state: AgentState) -> AgentState:
    clarifying_question = _coerce_str(state.get("clarifying_question")).strip()
    if bool(state.get("needs_clarification")) and clarifying_question:
        response_text = clarifying_question
    else:
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
        follow_ups = _extract_llm_follow_up_suggestions(result)
        if follow_ups:
            numbered = "\n".join(
                f"{index}. {item}" for index, item in enumerate(follow_ups, start=1)
            )
            response_text = f"{response_text}\n\n다음에 해볼 수 있는 분석\n{numbered}"
        updated_brief = _update_analysis_brief(state, follow_ups)
        conversation = list(state.get("conversation") or [])
        conversation.append({"role": "assistant", "content": response_text})
        return {
            "response_text": response_text,
            "conversation": _clip_conversation(conversation),
            "analysis_brief": updated_brief,
        }

    parts: list[str] = []
    error = state.get("error")
    if error:
        parts.append(f":x: SQL 생성 중 오류가 발생했습니다: {error}")
    elif explanation.strip():
        parts.append(explanation.strip())

    parts.extend(_build_sql_result_sections(state))

    if not parts:
        parts.append("요청을 처리할 수 있는 SQL 컨텍스트를 찾지 못했습니다.")
    follow_ups = _extract_llm_follow_up_suggestions(result)
    if follow_ups:
        numbered = "\n".join(f"{index}. {item}" for index, item in enumerate(follow_ups, start=1))
        parts.append(f"다음에 해볼 수 있는 분석\n{numbered}")
    response_text = "\n\n".join(parts)
    updated_brief = _update_analysis_brief(state, follow_ups)

    conversation = list(state.get("conversation") or [])
    conversation.append({"role": "assistant", "content": response_text})
    return {
        "response_text": response_text,
        "conversation": _clip_conversation(conversation),
        "analysis_brief": updated_brief,
    }
