"""Loop-based runtime for BigQuery agent.

This runtime keeps graph topology minimal and delegates behavior to an agent loop node
that invokes tool-like operations in-process.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from importlib import import_module
from typing import Any, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from data_bolt.tasks.bigquery.tools import guarded_execute_tool

from . import nodes
from .types import AgentState

LoopGraphBuilder = StateGraph[AgentState, None, Any, Any]
CompiledLoopGraph = CompiledStateGraph[AgentState, None, Any, Any]


class MemoryRuntimeCache(TypedDict, total=False):
    graph: CompiledLoopGraph
    saver: InMemorySaver


_memory_runtime_cache: MemoryRuntimeCache = {}
_postgres_graph_cache: dict[str, CompiledLoopGraph] = {}
_postgres_context_cache: dict[str, AbstractContextManager[Any]] = {}
_postgres_setup_done: set[str] = set()
_dynamodb_graph_cache: dict[tuple[str, str, str], CompiledLoopGraph] = {}


def _get_loop_max_steps() -> int:
    raw = os.getenv("BIGQUERY_AGENT_LOOP_MAX_STEPS", "4")
    try:
        value = int(raw)
    except ValueError:
        value = 4
    return max(1, min(8, value))


def _apply_patch(state: AgentState, updates: AgentState) -> None:
    state.update(updates)


def _run_guarded_execute(state: AgentState, *, action: str) -> AgentState:
    return cast(
        AgentState,
        guarded_execute_tool.run(
            action=action,
            candidate_sql=state.get("candidate_sql"),
            dry_run=state.get("dry_run"),
            pending_execution_sql=state.get("pending_execution_sql"),
            pending_execution_dry_run=state.get("pending_execution_dry_run"),
        ),
    )


def _node_agent_loop(state: AgentState) -> AgentState:
    working: AgentState = cast(AgentState, dict(state))
    working["runtime_mode"] = "loop"

    _apply_patch(working, nodes._node_reset_turn_state(working))
    _apply_patch(working, nodes._node_classify_relevance(working))
    if not working.get("should_respond"):
        _apply_patch(working, nodes._node_clear_response(working))
        return cast(AgentState, working)

    _apply_patch(working, nodes._node_ingest(working))

    step = 0
    max_steps = _get_loop_max_steps()
    while step < max_steps:
        step += 1
        _apply_patch(working, nodes._node_plan_turn_action(working))
        action = nodes._coerce_turn_action(working.get("action"), default="chat_reply")

        if action == "chat_reply":
            _apply_patch(working, nodes._node_chat_reply(working))
            break

        if action == "schema_lookup":
            _apply_patch(working, nodes._node_schema_lookup(working))
            break

        if action == "sql_validate_explain":
            _apply_patch(working, nodes._node_sql_validate_explain(working))
            break

        if action == "sql_generate":
            _apply_patch(working, nodes._node_sql_generate(working))
            _apply_patch(working, nodes._node_validate_candidate_sql(working))
            _apply_patch(working, _run_guarded_execute(working, action=action))
            break

        if action == "sql_execute":
            _apply_patch(working, nodes._node_sql_execute(working))
            _apply_patch(working, nodes._node_validate_candidate_sql(working))
            _apply_patch(working, _run_guarded_execute(working, action=action))
            break

        if action in {"execution_approve", "execution_cancel"}:
            _apply_patch(working, _run_guarded_execute(working, action=action))
            break

        # Safe fallback for unknown actions.
        working["action"] = "chat_reply"
        _apply_patch(working, nodes._node_chat_reply(working))
        break

    _apply_patch(working, nodes._node_compose_response(working))
    return cast(AgentState, working)


def _build_graph() -> LoopGraphBuilder:
    graph = StateGraph(AgentState)
    graph.add_node("agent_loop", _node_agent_loop)
    graph.add_edge(START, "agent_loop")
    graph.add_edge("agent_loop", END)
    return graph


def _invoke_with_memory(input_state: AgentState, thread_id: str) -> AgentState:
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


def _get_postgres_graph(conn_string: str) -> CompiledLoopGraph:
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


def _invoke_with_postgres(input_state: AgentState, thread_id: str) -> AgentState:
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
    dynamodb_saver = module.DynamoDBSaver

    return dynamodb_saver(
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
) -> CompiledLoopGraph:
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


def _invoke_with_dynamodb(input_state: AgentState, thread_id: str) -> AgentState:
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
