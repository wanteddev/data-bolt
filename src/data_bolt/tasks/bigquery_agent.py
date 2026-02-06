"""LangGraph-based orchestrator for BigQuery conversational tasks."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from data_bolt.tasks.bigquery_sql import (
    build_bigquery_sql,
    dry_run_bigquery_sql,
    execute_bigquery_sql,
    extract_sql_blocks,
)
from data_bolt.tasks.relevance import should_respond_to_message

logger = logging.getLogger(__name__)

_CONVERSATION_CLIP_LIMIT = int(os.getenv("BIGQUERY_AGENT_CONVERSATION_CLIP_LIMIT", "20"))


Intent = Literal[
    "ignore",
    "schema_lookup",
    "text_to_sql",
    "validate_sql",
    "execute_sql",
    "analysis_followup",
]


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


_memory_graph: Any = None
_memory_checkpointer: InMemorySaver | None = None
_postgres_setup_done = False


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


def _classify_intent(text: str, has_sql: bool) -> Intent:
    lowered = text.lower()
    if any(word in lowered for word in ("실행", "run", "execute", "돌려", "조회해줘")):
        return "execute_sql"
    if has_sql and any(word in lowered for word in ("검증", "dry", "비용", "validate", "오류")):
        return "validate_sql"
    if any(word in lowered for word in ("스키마", "schema", "table", "컬럼")):
        return "schema_lookup"
    if has_sql and any(
        word in lowered for word in ("분석", "해석", "설명", "optimize", "느려", "왜")
    ):
        return "analysis_followup"
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
    return {"should_respond": False, "intent": "ignore"}


def _node_classify_intent(state: AgentState) -> AgentState:
    text = state.get("text", "")
    user_sql = _extract_user_sql(text)
    return {"user_sql": user_sql, "intent": _classify_intent(text, bool(user_sql))}


def _node_generate_or_validate(state: AgentState) -> AgentState:
    text = state.get("text", "")
    intent = state.get("intent", "text_to_sql")
    user_sql = state.get("user_sql")

    if intent in {"validate_sql", "execute_sql"} and isinstance(user_sql, str):
        dry_run = dry_run_bigquery_sql(user_sql)
        return {
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
        "generation_result": result,
        "candidate_sql": sql,
        "dry_run": dry_run_data,
        "error": result.get("error") if isinstance(result.get("error"), str) else None,
    }


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


def _node_compose_response(state: AgentState) -> AgentState:
    if not state.get("should_respond"):
        return {"response_text": ""}

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


def _route_execution(state: AgentState) -> str:
    intent = state.get("intent", "text_to_sql")
    if intent == "execute_sql" and state.get("can_execute"):
        return "execute"
    return "skip"


def _build_graph() -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("ingest", _node_ingest)
    graph.add_node("classify_relevance", _node_classify_relevance)
    graph.add_node("classify_intent", _node_classify_intent)
    graph.add_node("generate_or_validate", _node_generate_or_validate)
    graph.add_node("policy_gate", _node_policy_gate)
    graph.add_node("execute_sql", _node_execute)
    graph.add_node("compose_response", _node_compose_response)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "classify_relevance")
    graph.add_conditional_edges(
        "classify_relevance",
        _route_relevance,
        {"proceed": "classify_intent", "stop": END},
    )
    graph.add_edge("classify_intent", "generate_or_validate")
    graph.add_edge("generate_or_validate", "policy_gate")
    graph.add_conditional_edges(
        "policy_gate",
        _route_execution,
        {"execute": "execute_sql", "skip": "compose_response"},
    )
    graph.add_edge("execute_sql", "compose_response")
    graph.add_edge("compose_response", END)
    return graph


def _invoke_graph_with_memory(input_state: AgentState, thread_id: str) -> AgentState:
    global _memory_graph, _memory_checkpointer

    if _memory_checkpointer is None:
        _memory_checkpointer = InMemorySaver()
    if _memory_graph is None:
        _memory_graph = _build_graph().compile(checkpointer=_memory_checkpointer)

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    graph_any = cast(Any, _memory_graph)
    output = graph_any.invoke(input_state, config=config)
    return cast(AgentState, output)


def _ensure_postgres_setup(conn_string: str) -> None:
    global _postgres_setup_done
    if _postgres_setup_done:
        return

    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(conn_string) as saver:
        saver.setup()
    _postgres_setup_done = True


def _invoke_graph_with_postgres(input_state: AgentState, thread_id: str) -> AgentState:
    from langgraph.checkpoint.postgres import PostgresSaver

    conn_string = os.getenv("LANGGRAPH_POSTGRES_URI")
    if not conn_string:
        raise ValueError("LANGGRAPH_POSTGRES_URI is not set")

    _ensure_postgres_setup(conn_string)

    with PostgresSaver.from_conn_string(conn_string) as saver:
        graph = _build_graph().compile(checkpointer=saver)
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
    if backend not in {"memory", "postgres"}:
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
    """
    Run LangGraph orchestration for BigQuery assistant flow.

    This function is synchronous and can be called from runtime adapters
    via `anyio.to_thread.run_sync`.
    """
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
            if not _is_checkpoint_backend_error(e):
                raise
            logger.warning("bigquery_agent.postgres_fallback_to_memory", extra={"error": str(e)})
            resolved_backend = "memory"
            state = _invoke_graph_with_memory(input_state, input_data.thread_id)
    else:
        state = _invoke_graph_with_memory(input_state, input_data.thread_id)

    generation_result = state.get("generation_result") or {}
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
    }
