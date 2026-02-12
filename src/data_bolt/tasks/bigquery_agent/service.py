"""Public service entrypoint for BigQuery agent."""

from __future__ import annotations

import logging
import os

from data_bolt.tasks.bigquery import tools as bigquery_tools

from . import checkpoint, loop_runtime, nodes
from .types import AgentInput, AgentPayload, AgentResult, AgentState

logger = logging.getLogger(__name__)

# Rebindable aliases for tests.
build_bigquery_sql = nodes.build_bigquery_sql
plan_turn_action = nodes.plan_turn_action
dry_run_bigquery_sql = nodes.dry_run_bigquery_sql
execute_bigquery_sql = nodes.execute_bigquery_sql
explain_schema_lookup = nodes.explain_schema_lookup
explain_sql_validation = nodes.explain_sql_validation
plan_free_chat = nodes.plan_free_chat
summarize_execution_result = nodes.summarize_execution_result

_build_graph = checkpoint._build_graph
_ensure_postgres_setup = checkpoint._ensure_postgres_setup
_build_dynamodb_saver = checkpoint._build_dynamodb_saver

_memory_runtime_cache = checkpoint._memory_runtime_cache
_postgres_graph_cache = checkpoint._postgres_graph_cache
_postgres_context_cache = checkpoint._postgres_context_cache
_postgres_setup_done = checkpoint._postgres_setup_done
_dynamodb_graph_cache = checkpoint._dynamodb_graph_cache

_loop_memory_runtime_cache = loop_runtime._memory_runtime_cache
_loop_postgres_graph_cache = loop_runtime._postgres_graph_cache
_loop_postgres_context_cache = loop_runtime._postgres_context_cache
_loop_postgres_setup_done = loop_runtime._postgres_setup_done
_loop_dynamodb_graph_cache = loop_runtime._dynamodb_graph_cache


def _resolve_runtime_mode() -> str:
    mode = nodes._coerce_str(os.getenv("BIGQUERY_AGENT_RUNTIME_MODE", "loop"), "loop").lower()
    if mode not in {"loop", "graph"}:
        return "loop"
    return mode


def _sync_node_dependencies() -> None:
    nodes.build_bigquery_sql = build_bigquery_sql
    nodes.plan_turn_action = plan_turn_action
    nodes.dry_run_bigquery_sql = dry_run_bigquery_sql
    nodes.execute_bigquery_sql = execute_bigquery_sql
    nodes.explain_schema_lookup = explain_schema_lookup
    nodes.explain_sql_validation = explain_sql_validation
    nodes.plan_free_chat = plan_free_chat
    nodes.summarize_execution_result = summarize_execution_result


def _sync_tool_dependencies() -> None:
    bigquery_tools._dry_run_callable = dry_run_bigquery_sql
    bigquery_tools._execute_callable = execute_bigquery_sql


def _sync_checkpoint_dependencies() -> None:
    checkpoint._build_graph = _build_graph
    checkpoint._ensure_postgres_setup = _ensure_postgres_setup
    checkpoint._build_dynamodb_saver = _build_dynamodb_saver


def _invoke_graph_with_memory(input_state: AgentState, thread_id: str) -> AgentState:
    _sync_checkpoint_dependencies()
    return checkpoint._invoke_graph_with_memory(input_state, thread_id)


def _invoke_graph_with_postgres(input_state: AgentState, thread_id: str) -> AgentState:
    _sync_checkpoint_dependencies()
    return checkpoint._invoke_graph_with_postgres(input_state, thread_id)


def _invoke_graph_with_dynamodb(input_state: AgentState, thread_id: str) -> AgentState:
    _sync_checkpoint_dependencies()
    return checkpoint._invoke_graph_with_dynamodb(input_state, thread_id)


def _is_checkpoint_backend_error(error: Exception) -> bool:
    return checkpoint._is_checkpoint_backend_error(error)


def _invoke_loop_with_memory(input_state: AgentState, thread_id: str) -> AgentState:
    return loop_runtime._invoke_with_memory(input_state, thread_id)


def _invoke_loop_with_postgres(input_state: AgentState, thread_id: str) -> AgentState:
    return loop_runtime._invoke_with_postgres(input_state, thread_id)


def _invoke_loop_with_dynamodb(input_state: AgentState, thread_id: str) -> AgentState:
    return loop_runtime._invoke_with_dynamodb(input_state, thread_id)


def _is_loop_checkpoint_backend_error(error: Exception) -> bool:
    return loop_runtime._is_checkpoint_backend_error(error)


def _build_thread_id(payload: AgentPayload) -> str:
    team_id = nodes._coerce_str(payload.get("team_id"), "local")
    channel_id = nodes._coerce_str(payload.get("channel_id"), "unknown")
    thread_ts = nodes._coerce_str(payload.get("thread_ts") or payload.get("message_ts"), "root")
    return f"{team_id}:{channel_id}:{thread_ts}"


def _normalize_payload(payload: AgentPayload) -> AgentInput:
    text = nodes._coerce_str(payload.get("text") or payload.get("question"), "").strip()
    thread_id = nodes._coerce_str(payload.get("thread_id") or _build_thread_id(payload))
    backend = nodes._coerce_str(
        os.getenv("LANGGRAPH_CHECKPOINT_BACKEND", "memory"), "memory"
    ).lower()
    if backend not in {"memory", "postgres", "dynamodb"}:
        backend = "memory"

    thread_ts = nodes._coerce_str(payload.get("thread_ts") or payload.get("message_ts"), "")
    return AgentInput(
        text=text,
        thread_id=thread_id,
        backend=backend,
        channel_type=nodes._coerce_str(payload.get("channel_type"), ""),
        is_mention=nodes._coerce_bool(payload.get("is_mention"), False),
        is_thread_followup=nodes._coerce_bool(payload.get("is_thread_followup"), False),
        team_id=nodes._coerce_str(payload.get("team_id"), "local"),
        channel_id=nodes._coerce_str(payload.get("channel_id"), ""),
        thread_ts=thread_ts,
        table_info=nodes._coerce_str(payload.get("table_info"), ""),
        glossary_info=nodes._coerce_str(payload.get("glossary_info"), ""),
        history=nodes._normalize_history(payload.get("history")),
    )


def _invoke_runtime(
    *,
    runtime_mode: str,
    input_state: AgentState,
    backend: str,
    thread_id: str,
) -> tuple[AgentState, str]:
    resolved_backend = backend

    if runtime_mode == "graph":
        if backend == "postgres":
            try:
                return _invoke_graph_with_postgres(input_state, thread_id), resolved_backend
            except Exception as error:
                allow_memory_fallback = nodes._env_truthy(
                    os.getenv("LANGGRAPH_POSTGRES_FALLBACK_TO_MEMORY"), False
                )
                if not allow_memory_fallback or not _is_checkpoint_backend_error(error):
                    raise
                logger.warning(
                    "bigquery_agent.postgres_fallback_to_memory",
                    extra={
                        "error": str(error),
                        "thread_id": thread_id,
                        "runtime_mode": runtime_mode,
                    },
                )
                resolved_backend = "memory"
                return _invoke_graph_with_memory(input_state, thread_id), resolved_backend
        if backend == "dynamodb":
            return _invoke_graph_with_dynamodb(input_state, thread_id), resolved_backend
        return _invoke_graph_with_memory(input_state, thread_id), resolved_backend

    if backend == "postgres":
        try:
            return _invoke_loop_with_postgres(input_state, thread_id), resolved_backend
        except Exception as error:
            allow_memory_fallback = nodes._env_truthy(
                os.getenv("LANGGRAPH_POSTGRES_FALLBACK_TO_MEMORY"), False
            )
            if not allow_memory_fallback or not _is_loop_checkpoint_backend_error(error):
                raise
            logger.warning(
                "bigquery_agent.loop_postgres_fallback_to_memory",
                extra={"error": str(error), "thread_id": thread_id, "runtime_mode": runtime_mode},
            )
            resolved_backend = "memory"
            return _invoke_loop_with_memory(input_state, thread_id), resolved_backend
    if backend == "dynamodb":
        return _invoke_loop_with_dynamodb(input_state, thread_id), resolved_backend
    return _invoke_loop_with_memory(input_state, thread_id), resolved_backend


def run_bigquery_agent(payload: AgentPayload) -> AgentResult:
    _sync_node_dependencies()
    _sync_tool_dependencies()
    input_data = _normalize_payload(payload)
    runtime_mode = _resolve_runtime_mode()

    input_state: AgentState = {
        "runtime_mode": runtime_mode,
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

    state, resolved_backend = _invoke_runtime(
        runtime_mode=runtime_mode,
        input_state=input_state,
        backend=input_data.backend,
        thread_id=input_data.thread_id,
    )

    generation_result = state.get("generation_result") or {}
    generation_result_meta = generation_result.get("meta")
    generation_result["meta"] = (
        generation_result_meta if isinstance(generation_result_meta, dict) else {}
    )
    generation_result["meta"]["routing"] = nodes._build_routing_meta(state)
    response_text = state.get("response_text") or ""
    validation = state.get("dry_run") or generation_result.get("validation") or {}

    return {
        "thread_id": input_data.thread_id,
        "backend": resolved_backend,
        "action": state.get("action"),
        "should_respond": bool(state.get("should_respond")),
        "response_text": response_text,
        "candidate_sql": state.get("candidate_sql"),
        "validation": validation,
        "execution": state.get("execution") or {},
        "generation_result": generation_result,
        "conversation_turns": len(state.get("conversation") or []),
        "routing": nodes._build_routing_meta(state),
    }
