"""LangGraph-based orchestrator for BigQuery conversational tasks."""

from __future__ import annotations

import json
import logging
import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, NotRequired, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from data_bolt.tasks.bigquery_sql import (
    build_bigquery_sql,
    classify_intent_with_laas,
    dry_run_bigquery_sql,
    execute_bigquery_sql,
    extract_sql_blocks,
    plan_free_chat_with_laas,
)
from data_bolt.tasks.relevance import looks_like_data_request, should_respond_to_message

logger = logging.getLogger(__name__)

_CONVERSATION_CLIP_LIMIT = int(os.getenv("BIGQUERY_AGENT_CONVERSATION_CLIP_LIMIT", "20"))
_ALLOWED_ACTIONS = {
    "text_to_sql",
    "validate_sql",
    "execute_sql",
    "schema_lookup",
    "analysis_followup",
}

Intent = Literal[
    "ignore",
    "chat",
    "schema_lookup",
    "text_to_sql",
    "validate_sql",
    "execute_sql",
    "analysis_followup",
    "data_workflow",
    "free_chat",
]
Route = Literal["data", "chat"]


class ConversationMessage(TypedDict):
    role: str
    content: str


class AgentPayload(TypedDict):
    user_id: NotRequired[str]
    text: NotRequired[str]
    question: NotRequired[str]
    channel_type: NotRequired[str]
    is_mention: NotRequired[bool]
    is_thread_followup: NotRequired[bool]
    team_id: NotRequired[str]
    channel_id: NotRequired[str]
    thread_ts: NotRequired[str]
    message_ts: NotRequired[str]
    table_info: NotRequired[str]
    glossary_info: NotRequired[str]
    history: NotRequired[list[ConversationMessage]]
    thread_id: NotRequired[str]
    response_url: NotRequired[str]
    include_thread_history: NotRequired[bool]


class AgentResult(TypedDict):
    thread_id: str
    backend: str
    intent: Intent | None
    should_respond: bool
    response_text: str
    candidate_sql: str | None
    validation: dict[str, Any]
    execution: dict[str, Any]
    generation_result: dict[str, Any]
    conversation_turns: int
    routing: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentInput:
    text: str
    thread_id: str
    backend: str
    channel_type: str
    is_mention: bool
    is_thread_followup: bool
    team_id: str
    channel_id: str
    thread_ts: str
    table_info: str
    glossary_info: str
    history: list[ConversationMessage]


class AgentState(TypedDict, total=False):
    text: str
    channel_type: str
    is_mention: bool
    is_thread_followup: bool
    team_id: str
    channel_id: str
    thread_ts: str
    table_info: str
    glossary_info: str
    history: list[ConversationMessage]
    conversation: list[ConversationMessage]
    should_respond: bool
    intent: Intent
    user_sql: str | None
    candidate_sql: str | None
    dry_run: dict[str, Any]
    generation_result: dict[str, Any]
    can_execute: bool
    execution: dict[str, Any]
    response_text: str
    error: str | None
    route: Route
    intent_confidence: float
    intent_reason: str
    planned_actions: list[str]
    chat_result: dict[str, Any]
    fallback_used: bool
    planner_reason: str


AgentGraphBuilder = StateGraph[AgentState, None, Any, Any]
CompiledAgentGraph = CompiledStateGraph[AgentState, None, Any, Any]


class MemoryRuntimeCache(TypedDict, total=False):
    graph: CompiledAgentGraph
    saver: InMemorySaver


_memory_runtime_cache: MemoryRuntimeCache = {}
_postgres_graph_cache: dict[str, CompiledAgentGraph] = {}
_postgres_context_cache: dict[str, AbstractContextManager[Any]] = {}
_postgres_setup_done: set[str] = set()
_dynamodb_graph_cache: dict[tuple[str, str, str], CompiledAgentGraph] = {}


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


def _classify_intent_rule(text: str, has_sql: bool) -> Intent:
    lowered = text.lower()
    if any(word in lowered for word in ("실행", "run", "execute", "돌려", "조회해줘")):
        return "execute_sql"
    if has_sql and any(word in lowered for word in ("검증", "dry", "비용", "validate", "오류")):
        return "validate_sql"
    if any(
        word in lowered
        for word in ("스키마", "schema", "table", "컬럼", "column", "db", "database")
    ):
        return "schema_lookup"
    if has_sql and any(
        word in lowered for word in ("분석", "해석", "설명", "optimize", "느려", "왜")
    ):
        return "analysis_followup"
    if looks_like_data_request(text):
        return "text_to_sql"
    if any(word in lowered for word in ("조회", "집계", "몇명", "사용자 수", "count", "통계")):
        return "text_to_sql"
    if not has_sql and not looks_like_data_request(text):
        return "chat"
    return "text_to_sql"


def _clip_conversation(conversation: list[ConversationMessage]) -> list[ConversationMessage]:
    if _CONVERSATION_CLIP_LIMIT > 0 and len(conversation) > _CONVERSATION_CLIP_LIMIT:
        return conversation[-_CONVERSATION_CLIP_LIMIT:]
    return conversation


def _extract_sql_from_generation(result: dict[str, Any]) -> str | None:
    structured = result.get("answer_structured") or {}
    sql = structured.get("sql")
    if isinstance(sql, str) and sql.strip():
        return sql.strip()

    choices = result.get("choices") or []
    if choices:
        content = (choices[0] or {}).get("message", {}).get("content")
        if isinstance(content, str) and content.strip():
            blocks = extract_sql_blocks(content)
            if blocks:
                return blocks[0]
            return content.strip()
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


def _coerce_actions(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = item.strip().lower()
        if not normalized or normalized == "none":
            continue
        if normalized in _ALLOWED_ACTIONS:
            out.append(normalized)
    return out


def _build_routing_meta(state: AgentState) -> dict[str, Any]:
    return {
        "intent": state.get("intent") or "",
        "confidence": state.get("intent_confidence", 0.0),
        "reason": state.get("intent_reason") or "",
        "actions": state.get("planned_actions") or [],
        "route": state.get("route") or "",
        "fallback_used": bool(state.get("fallback_used")),
        "planner_reason": state.get("planner_reason") or "",
        "prompt_version": "v1",
    }


def _node_reset_turn_state(state: AgentState) -> AgentState:
    return {
        "text": (state.get("text") or "").strip(),
        "should_respond": False,
        "intent": "ignore",
        "user_sql": None,
        "candidate_sql": None,
        "dry_run": {},
        "generation_result": {},
        "can_execute": False,
        "execution": {},
        "response_text": "",
        "error": None,
        "route": "chat",
        "intent_confidence": 0.0,
        "intent_reason": "",
        "planned_actions": [],
        "chat_result": {},
        "fallback_used": False,
        "planner_reason": "",
    }


def _node_clear_response(state: AgentState) -> AgentState:
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
    return {"should_respond": False, "intent": "ignore", "route": "chat"}


def _node_classify_intent_llm(state: AgentState) -> AgentState:
    text = state.get("text", "")
    user_sql = _extract_user_sql(text)
    llm_enabled = _env_truthy(os.getenv("BIGQUERY_INTENT_LLM_ENABLED"), True)
    confidence_threshold = float(os.getenv("BIGQUERY_INTENT_CONFIDENCE_THRESHOLD", "0.55"))
    fallback_used = False

    if llm_enabled:
        try:
            raw = classify_intent_with_laas(
                text=text,
                history=cast(list[dict[str, Any]], state.get("conversation") or []),
                channel_type=state.get("channel_type", ""),
                is_mention=bool(state.get("is_mention")),
                is_thread_followup=bool(state.get("is_thread_followup")),
            )
            raw_intent = _coerce_str(raw.get("intent"), "free_chat").lower()
            confidence = float(raw.get("confidence", 0.0))
            reason = _coerce_str(raw.get("reason"), "llm routing")
            actions = _coerce_actions(raw.get("actions"))
            route: Route = "data" if raw_intent == "data_workflow" else "chat"
            intent: Intent = "data_workflow" if route == "data" else "free_chat"
            if route == "data" and confidence < confidence_threshold:
                route = "chat"
                intent = "free_chat"
                reason = f"low confidence fallback: {reason}"
            return {
                "user_sql": user_sql,
                "intent": intent,
                "route": route,
                "intent_confidence": confidence,
                "intent_reason": reason,
                "planned_actions": actions,
                "fallback_used": False,
            }
        except Exception as e:
            logger.warning("bigquery_agent.intent_llm_fallback", extra={"error": str(e)})
            fallback_used = True

    rule_intent = _classify_intent_rule(text, bool(user_sql))
    route = "chat" if rule_intent == "chat" else "data"
    routed_intent: Intent = "free_chat" if route == "chat" else "data_workflow"
    fallback_actions: list[str] = [] if route == "chat" else [rule_intent]
    return {
        "user_sql": user_sql,
        "intent": routed_intent,
        "route": route,
        "intent_confidence": 0.51 if route == "data" else 0.49,
        "intent_reason": "rule-based fallback",
        "planned_actions": fallback_actions,
        "fallback_used": fallback_used or not llm_enabled,
    }


def _node_free_chat_planner(state: AgentState) -> AgentState:
    max_actions = max(1, int(os.getenv("BIGQUERY_CHAT_PLANNER_MAX_ACTIONS", "2")))
    allow_execute = _env_truthy(os.getenv("BIGQUERY_CHAT_ALLOW_EXECUTE_IN_CHAT"), False)

    planned: dict[str, Any] = {}
    planner_reason = ""
    actions: list[str] = []
    try:
        planned = plan_free_chat_with_laas(
            text=state.get("text", ""),
            history=cast(list[dict[str, Any]], state.get("conversation") or []),
            allow_execute_in_chat=allow_execute,
            max_actions=max_actions,
        )
        actions = _coerce_actions(planned.get("actions"))
        if not allow_execute:
            actions = [action for action in actions if action != "execute_sql"]
        actions = actions[:max_actions]
        planner_reason = _coerce_str(planned.get("action_reason"), "")
    except Exception as e:
        logger.warning("bigquery_agent.free_chat_planner_failed", extra={"error": str(e)})

    assistant_response = _coerce_str(planned.get("assistant_response"), "")
    if not assistant_response:
        assistant_response = "좋아요. 맥락을 이어서 도와드릴게요. 필요한 조건을 알려주세요."

    normalized_intent: Intent = "chat"
    if actions:
        first_action = actions[0]
        if first_action in _ALLOWED_ACTIONS:
            normalized_intent = cast(Intent, first_action)

    chat_result = {
        "assistant_response": assistant_response,
        "actions": actions,
        "action_reason": planner_reason,
    }
    return {
        "intent": normalized_intent,
        "planned_actions": actions,
        "chat_result": chat_result,
        "planner_reason": planner_reason,
    }


def _node_data_generate_or_validate(state: AgentState) -> AgentState:
    text = state.get("text", "")
    intent = state.get("intent", "text_to_sql")
    if intent == "data_workflow":
        actions = state.get("planned_actions") or []
        first_action = actions[0] if actions else "text_to_sql"
        intent = cast(Intent, first_action if first_action in _ALLOWED_ACTIONS else "text_to_sql")
    user_sql = state.get("user_sql")

    if intent in {"validate_sql", "execute_sql"} and isinstance(user_sql, str):
        dry_run = dry_run_bigquery_sql(user_sql)
        return {
            "intent": intent,
            "candidate_sql": user_sql,
            "dry_run": dry_run,
            "generation_result": {
                "answer_structured": {
                    "sql": user_sql,
                    "explanation": "요청에 포함된 SQL을 검증했습니다.",
                    "instruction_type": "direct_sql_validation",
                }
            },
        }

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

    result = build_bigquery_sql(payload)
    sql = _extract_sql_from_generation(result)
    validation = result.get("validation")
    dry_run_data: dict[str, Any] = validation if isinstance(validation, dict) else {}

    return {
        "intent": intent,
        "generation_result": result,
        "candidate_sql": sql,
        "dry_run": dry_run_data,
        "error": result.get("error") if isinstance(result.get("error"), str) else None,
    }


def _node_validate_candidate_sql(state: AgentState) -> AgentState:
    sql = state.get("candidate_sql")
    if not isinstance(sql, str) or not sql.strip():
        return {}

    dry_run = state.get("dry_run")
    if isinstance(dry_run, dict) and "success" in dry_run:
        return {}

    return {"dry_run": dry_run_bigquery_sql(sql)}


def _node_policy_gate(state: AgentState) -> AgentState:
    intent = state.get("intent", "text_to_sql")
    if intent != "execute_sql":
        return {"can_execute": False}

    sql = state.get("candidate_sql")
    if not sql:
        return {
            "can_execute": False,
            "execution": {"success": False, "error": "실행할 SQL이 없습니다."},
        }

    dry_run = state.get("dry_run") or {}
    if dry_run and not dry_run.get("success", True):
        return {
            "can_execute": False,
            "execution": {"success": False, "error": "dry-run 실패로 실행을 중단했습니다."},
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
        }

    return {"can_execute": True}


def _node_execute(state: AgentState) -> AgentState:
    if not state.get("can_execute"):
        return {}
    sql = state.get("candidate_sql")
    if not sql:
        return {"execution": {"success": False, "error": "실행할 SQL이 없습니다."}}
    return {"execution": execute_bigquery_sql(sql)}


def _node_free_chat_respond(state: AgentState) -> AgentState:
    chat_result_raw = state.get("chat_result")
    chat_result = chat_result_raw if isinstance(chat_result_raw, dict) else {}
    assistant_response = _coerce_str(chat_result.get("assistant_response"), "")
    parts: list[str] = [assistant_response] if assistant_response else []

    if state.get("candidate_sql"):
        parts.append(f"```sql\n{state.get('candidate_sql')}\n```")

    dry_run = state.get("dry_run") or {}
    if dry_run:
        if dry_run.get("success"):
            parts.append(
                ":white_check_mark: Dry-run 통과"
                f"\n- bytes processed: {_format_number(dry_run.get('total_bytes_processed'))}"
                f"\n- estimated cost (USD): {dry_run.get('estimated_cost_usd', '-')}"
            )
        else:
            parts.append(f":warning: Dry-run 실패\n- error: {dry_run.get('error', '-')}")

    execution = state.get("execution") or {}
    if execution:
        if execution.get("success"):
            preview = execution.get("preview_rows")
            preview_text = json.dumps(preview, ensure_ascii=False, default=str)[:1500]
            parts.append(
                ":rocket: 쿼리 실행 완료"
                f"\n- job_id: {execution.get('job_id', '-')}"
                f"\n- row_count: {_format_number(execution.get('row_count'))}"
                f"\n- preview: `{preview_text}`"
            )
        else:
            parts.append(f":warning: 쿼리 실행 생략/실패\n- reason: {execution.get('error', '-')}")

    response_text = "\n\n".join(part for part in parts if part.strip())
    if not response_text:
        response_text = "추가로 확인하고 싶은 내용을 알려주세요."
    return {"response_text": response_text}


def _node_compose_response(state: AgentState) -> AgentState:
    if not state.get("should_respond"):
        return {"response_text": ""}

    route = state.get("route", "data")
    if route == "chat":
        response_text = _coerce_str(state.get("response_text"), "")
        conversation = list(state.get("conversation") or [])
        conversation.append({"role": "assistant", "content": response_text})
        return {"response_text": response_text, "conversation": _clip_conversation(conversation)}

    result_raw = state.get("generation_result")
    result = result_raw if isinstance(result_raw, dict) else {}
    structured_raw = result.get("answer_structured")
    structured = structured_raw if isinstance(structured_raw, dict) else {}
    explanation_raw = structured.get("explanation")
    explanation = explanation_raw if isinstance(explanation_raw, str) else ""

    parts: list[str] = []
    error = state.get("error")
    if error:
        parts.append(f":x: SQL 생성 중 오류가 발생했습니다: {error}")
    elif explanation.strip():
        parts.append(explanation.strip())

    sql = state.get("candidate_sql")
    if isinstance(sql, str) and sql.strip():
        parts.append(f"```sql\n{sql.strip()}\n```")

    dry_run = state.get("dry_run") or {}
    if dry_run:
        success = dry_run.get("success")
        bytes_processed = dry_run.get("total_bytes_processed")
        estimated_cost = dry_run.get("estimated_cost_usd")
        if success:
            parts.append(
                ":white_check_mark: Dry-run 통과"
                f"\n- bytes processed: {_format_number(bytes_processed)}"
                f"\n- estimated cost (USD): {estimated_cost if estimated_cost is not None else '-'}"
            )
        else:
            parts.append(f":warning: Dry-run 실패\n- error: {dry_run.get('error', '-')}")

    execution = state.get("execution") or {}
    if execution:
        if execution.get("success"):
            preview = execution.get("preview_rows")
            preview_text = json.dumps(preview, ensure_ascii=False, default=str)[:1500]
            parts.append(
                ":rocket: 쿼리 실행 완료"
                f"\n- job_id: {execution.get('job_id', '-')}"
                f"\n- row_count: {_format_number(execution.get('row_count'))}"
                f"\n- preview: `{preview_text}`"
            )
        else:
            parts.append(f":warning: 쿼리 실행 생략/실패\n- reason: {execution.get('error', '-')}")

    if not parts:
        parts.append("요청을 처리할 수 있는 SQL 컨텍스트를 찾지 못했습니다.")

    parts.append("필요하면 조건을 더 구체화해서 다시 요청해주세요.")
    response_text = "\n\n".join(parts)

    conversation = list(state.get("conversation") or [])
    conversation.append({"role": "assistant", "content": response_text})
    return {"response_text": response_text, "conversation": _clip_conversation(conversation)}


def _route_relevance(state: AgentState) -> str:
    return "proceed" if state.get("should_respond") else "stop"


def _route_intent(state: AgentState) -> str:
    return "route_data" if state.get("route") == "data" else "route_chat"


def _route_chat_plan(state: AgentState) -> str:
    actions = state.get("planned_actions") or []
    return "needs_data" if actions else "chat_only"


def _route_execution(state: AgentState) -> str:
    intent = state.get("intent", "text_to_sql")
    if intent == "execute_sql" and state.get("can_execute"):
        return "execute"
    return "skip_chat" if state.get("route") == "chat" else "skip_data"


def _route_after_policy(state: AgentState) -> str:
    return "chat" if state.get("route") == "chat" else "data"


def _build_graph() -> AgentGraphBuilder:
    graph = StateGraph(AgentState)
    graph.add_node("reset_turn_state", _node_reset_turn_state)
    graph.add_node("clear_response", _node_clear_response)
    graph.add_node("ingest", _node_ingest)
    graph.add_node("classify_relevance", _node_classify_relevance)
    graph.add_node("classify_intent_llm", _node_classify_intent_llm)
    graph.add_node("free_chat_planner", _node_free_chat_planner)
    graph.add_node("data_generate_or_validate", _node_data_generate_or_validate)
    graph.add_node("validate_candidate_sql", _node_validate_candidate_sql)
    graph.add_node("policy_gate", _node_policy_gate)
    graph.add_node("execute_sql", _node_execute)
    graph.add_node("free_chat_respond", _node_free_chat_respond)
    graph.add_node("compose_response", _node_compose_response)

    graph.add_edge(START, "reset_turn_state")
    graph.add_edge("reset_turn_state", "classify_relevance")
    graph.add_conditional_edges(
        "classify_relevance",
        _route_relevance,
        {"proceed": "ingest", "stop": "clear_response"},
    )
    graph.add_edge("ingest", "classify_intent_llm")
    graph.add_edge("clear_response", END)
    graph.add_conditional_edges(
        "classify_intent_llm",
        _route_intent,
        {"route_data": "data_generate_or_validate", "route_chat": "free_chat_planner"},
    )
    graph.add_conditional_edges(
        "free_chat_planner",
        _route_chat_plan,
        {"needs_data": "data_generate_or_validate", "chat_only": "free_chat_respond"},
    )
    graph.add_edge("data_generate_or_validate", "validate_candidate_sql")
    graph.add_edge("validate_candidate_sql", "policy_gate")
    graph.add_conditional_edges(
        "policy_gate",
        _route_execution,
        {
            "execute": "execute_sql",
            "skip_chat": "free_chat_respond",
            "skip_data": "compose_response",
        },
    )
    graph.add_conditional_edges(
        "execute_sql",
        _route_after_policy,
        {"chat": "free_chat_respond", "data": "compose_response"},
    )
    graph.add_conditional_edges(
        "free_chat_respond",
        _route_after_policy,
        {"chat": "compose_response", "data": "compose_response"},
    )
    graph.add_edge("compose_response", END)
    return graph


def _invoke_graph_with_memory(input_state: AgentState, thread_id: str) -> AgentState:
    graph = _memory_runtime_cache.get("graph")
    if graph is None:
        saver = InMemorySaver()
        graph = _build_graph().compile(checkpointer=saver)
        _memory_runtime_cache["graph"] = graph
        _memory_runtime_cache["saver"] = saver

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    graph_any = cast(Any, graph)
    output = graph_any.invoke(input_state, config=config)
    return cast(AgentState, output)


def _ensure_postgres_setup(conn_string: str) -> None:
    if conn_string in _postgres_setup_done:
        return

    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(conn_string) as saver:
        saver.setup()
    _postgres_setup_done.add(conn_string)


def _get_postgres_graph(conn_string: str) -> CompiledAgentGraph:
    graph = _postgres_graph_cache.get(conn_string)
    if graph is not None:
        return graph

    from langgraph.checkpoint.postgres import PostgresSaver

    context_manager = cast(AbstractContextManager[Any], PostgresSaver.from_conn_string(conn_string))
    saver = cast(Any, context_manager).__enter__()
    graph = _build_graph().compile(checkpointer=saver)
    _postgres_context_cache[conn_string] = context_manager
    _postgres_graph_cache[conn_string] = graph
    return graph


def _invoke_graph_with_postgres(input_state: AgentState, thread_id: str) -> AgentState:
    conn_string = os.getenv("LANGGRAPH_POSTGRES_URI")
    if not conn_string:
        raise ValueError("LANGGRAPH_POSTGRES_URI is not set")

    _ensure_postgres_setup(conn_string)

    graph = _get_postgres_graph(conn_string)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    graph_any = cast(Any, graph)
    output = graph_any.invoke(input_state, config=config)
    return cast(AgentState, output)


def _build_dynamodb_saver(
    *, table_name: str, region_name: str | None, endpoint_url: str | None
) -> Any:
    module = cast(Any, import_module("langgraph_checkpoint_aws"))
    DynamoDBSaver = module.DynamoDBSaver

    return DynamoDBSaver(
        table_name=table_name,
        region_name=region_name,
        endpoint_url=endpoint_url,
    )


def _resolve_dynamodb_config() -> tuple[str, str | None, str | None]:
    table_name = os.getenv("LANGGRAPH_DYNAMODB_TABLE")
    if not table_name:
        raise ValueError("LANGGRAPH_DYNAMODB_TABLE is not set")
    region_name = os.getenv("LANGGRAPH_DYNAMODB_REGION") or None
    endpoint_url = os.getenv("LANGGRAPH_DYNAMODB_ENDPOINT_URL") or None
    return table_name, region_name, endpoint_url


def _get_dynamodb_graph(
    table_name: str, region_name: str | None, endpoint_url: str | None
) -> CompiledAgentGraph:
    cache_key = (table_name, region_name or "", endpoint_url or "")
    graph = _dynamodb_graph_cache.get(cache_key)
    if graph is not None:
        return graph

    saver = _build_dynamodb_saver(
        table_name=table_name,
        region_name=region_name,
        endpoint_url=endpoint_url,
    )
    graph = _build_graph().compile(checkpointer=saver)
    _dynamodb_graph_cache[cache_key] = graph
    return graph


def _invoke_graph_with_dynamodb(input_state: AgentState, thread_id: str) -> AgentState:
    table_name, region_name, endpoint_url = _resolve_dynamodb_config()
    graph = _get_dynamodb_graph(table_name, region_name, endpoint_url)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    graph_any = cast(Any, graph)
    output = graph_any.invoke(input_state, config=config)
    return cast(AgentState, output)


def _is_checkpoint_backend_error(error: Exception) -> bool:
    if isinstance(error, ValueError) and "LANGGRAPH_POSTGRES_URI" in str(error):
        return True
    try:
        import psycopg
    except Exception:
        return False
    return isinstance(error, psycopg.Error)


def _build_thread_id(payload: AgentPayload) -> str:
    team_id = _coerce_str(payload.get("team_id"), "local")
    channel_id = _coerce_str(payload.get("channel_id"), "unknown")
    thread_ts = _coerce_str(payload.get("thread_ts") or payload.get("message_ts"), "root")
    return f"{team_id}:{channel_id}:{thread_ts}"


def _normalize_payload(payload: AgentPayload) -> AgentInput:
    text = _coerce_str(payload.get("text") or payload.get("question"), "").strip()
    thread_id = _coerce_str(payload.get("thread_id") or _build_thread_id(payload))
    backend = _coerce_str(os.getenv("LANGGRAPH_CHECKPOINT_BACKEND", "memory"), "memory").lower()
    if backend not in {"memory", "postgres", "dynamodb"}:
        backend = "memory"

    thread_ts = _coerce_str(payload.get("thread_ts") or payload.get("message_ts"), "")
    return AgentInput(
        text=text,
        thread_id=thread_id,
        backend=backend,
        channel_type=_coerce_str(payload.get("channel_type"), ""),
        is_mention=_coerce_bool(payload.get("is_mention"), False),
        is_thread_followup=_coerce_bool(payload.get("is_thread_followup"), False),
        team_id=_coerce_str(payload.get("team_id"), "local"),
        channel_id=_coerce_str(payload.get("channel_id"), ""),
        thread_ts=thread_ts,
        table_info=_coerce_str(payload.get("table_info"), ""),
        glossary_info=_coerce_str(payload.get("glossary_info"), ""),
        history=_normalize_history(payload.get("history")),
    )


def run_bigquery_agent(payload: AgentPayload) -> AgentResult:
    input_data = _normalize_payload(payload)

    input_state: AgentState = {
        "text": input_data.text,
        "channel_type": input_data.channel_type,
        "is_mention": input_data.is_mention,
        "is_thread_followup": input_data.is_thread_followup,
        "team_id": input_data.team_id,
        "channel_id": input_data.channel_id,
        "thread_ts": input_data.thread_ts,
        "table_info": input_data.table_info,
        "glossary_info": input_data.glossary_info,
        "history": input_data.history,
    }

    resolved_backend = input_data.backend
    if input_data.backend == "postgres":
        try:
            state = _invoke_graph_with_postgres(input_state, input_data.thread_id)
        except Exception as e:
            allow_memory_fallback = _env_truthy(
                os.getenv("LANGGRAPH_POSTGRES_FALLBACK_TO_MEMORY"), False
            )
            if not allow_memory_fallback or not _is_checkpoint_backend_error(e):
                raise
            logger.warning(
                "bigquery_agent.postgres_fallback_to_memory",
                extra={"error": str(e), "thread_id": input_data.thread_id},
            )
            resolved_backend = "memory"
            state = _invoke_graph_with_memory(input_state, input_data.thread_id)
    elif input_data.backend == "dynamodb":
        state = _invoke_graph_with_dynamodb(input_state, input_data.thread_id)
    else:
        state = _invoke_graph_with_memory(input_state, input_data.thread_id)

    generation_result = state.get("generation_result") or {}
    generation_result_meta = generation_result.get("meta")
    generation_result["meta"] = (
        generation_result_meta if isinstance(generation_result_meta, dict) else {}
    )
    generation_result["meta"]["routing"] = _build_routing_meta(state)
    response_text = state.get("response_text") or ""
    validation = state.get("dry_run") or generation_result.get("validation") or {}

    return {
        "thread_id": input_data.thread_id,
        "backend": resolved_backend,
        "intent": state.get("intent"),
        "should_respond": bool(state.get("should_respond")),
        "response_text": response_text,
        "candidate_sql": state.get("candidate_sql"),
        "validation": validation,
        "execution": state.get("execution") or {},
        "generation_result": generation_result,
        "conversation_turns": len(state.get("conversation") or []),
        "routing": _build_routing_meta(state),
    }
