"""Checkpoint backend integration for BigQuery agent graph."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from importlib import import_module
from typing import Any, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from . import graph as graph_module
from .graph import CompiledAgentGraph
from .types import AgentState


class MemoryRuntimeCache(TypedDict, total=False):
    graph: CompiledAgentGraph
    saver: InMemorySaver


_build_graph = graph_module._build_graph

_memory_runtime_cache: MemoryRuntimeCache = {}
_postgres_graph_cache: dict[str, CompiledAgentGraph] = {}
_postgres_context_cache: dict[str, AbstractContextManager[Any]] = {}
_postgres_setup_done: set[str] = set()
_dynamodb_graph_cache: dict[tuple[str, str, str], CompiledAgentGraph] = {}


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
