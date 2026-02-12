"""Loop-based runtime for BigQuery agent.

Agent(LLM planning) <-> tools(execution-only) loop runtime built on StateGraph.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from importlib import import_module
from typing import Any, TypedDict, cast
from uuid import uuid4

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


class ToolCall(TypedDict):
    id: str
    name: str
    args: dict[str, Any]


ToolExecutor = Callable[[AgentState, dict[str, Any]], AgentState]


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
    result = cast(
        AgentState,
        guarded_execute_tool.run(
            action=action,
            candidate_sql=state.get("candidate_sql"),
            dry_run=state.get("dry_run"),
            pending_execution_sql=state.get("pending_execution_sql"),
            pending_execution_dry_run=state.get("pending_execution_dry_run"),
        ),
    )
    result.pop("can_execute", None)
    return result


def _coerce_tool_calls(value: Any) -> list[ToolCall]:
    if not isinstance(value, list):
        return []
    tool_calls: list[ToolCall] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        args = item.get("args")
        call_id = item.get("id")
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(args, dict):
            args = {}
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"tc_{len(tool_calls) + 1}"
        tool_calls.append({"id": call_id, "name": name.strip(), "args": args})
    return tool_calls


def _resolve_sql_from_state(state: AgentState) -> str | None:
    user_sql = nodes._coerce_str(state.get("user_sql")).strip()
    if user_sql:
        return user_sql
    pending_sql = nodes._coerce_str(state.get("pending_execution_sql")).strip()
    if pending_sql:
        return pending_sql
    candidate_sql = nodes._coerce_str(state.get("candidate_sql")).strip()
    if candidate_sql:
        return candidate_sql
    last_sql = nodes._coerce_str(state.get("last_candidate_sql")).strip()
    if last_sql:
        return last_sql
    return None


def _request_id(state: AgentState) -> str:
    execution = state.get("execution")
    if isinstance(execution, dict):
        request = execution.get("request")
        if isinstance(request, dict):
            request_id = request.get("id")
            if isinstance(request_id, str) and request_id.strip():
                return request_id
    return f"exec-{uuid4().hex[:10]}"


def _ensure_turn_state(state: AgentState) -> None:
    turn = state.get("turn")
    max_steps = _get_loop_max_steps()
    if not isinstance(turn, dict):
        turn = {}
    step_raw = turn.get("step")
    try:
        step = int(step_raw) if step_raw is not None else 0
    except (TypeError, ValueError):
        step = 0
    max_steps_raw = turn.get("max_steps")
    try:
        max_steps_value = int(max_steps_raw) if max_steps_raw is not None else max_steps
    except (TypeError, ValueError):
        max_steps_value = max_steps
    turn.setdefault("terminated", False)
    turn.setdefault("termination_reason", "")
    turn["step"] = max(0, step)
    turn["max_steps"] = max(1, min(8, max_steps_value))
    state["turn"] = turn


def _tool_schema_lookup(state: AgentState, args: dict[str, Any]) -> AgentState:
    question = nodes._coerce_str(args.get("question") or state.get("text")).strip()
    table_info = nodes._coerce_str(state.get("table_info")).strip()
    glossary_info = nodes._coerce_str(state.get("glossary_info")).strip()
    history = cast(list[dict[str, Any]], state.get("conversation") or [])
    try:
        lookup = nodes.explain_schema_lookup(
            question=question,
            table_info=table_info,
            glossary_info=glossary_info,
            history=history,
        )
    except Exception:
        lookup = {
            "response_text": "현재 스키마 메타데이터가 비어 있습니다. 테이블/컬럼 또는 지표명을 함께 알려주세요.",
            "reference_sql": None,
            "meta": {},
        }
    response = nodes._coerce_str(lookup.get("response_text"), "").strip()
    reference_sql_raw = lookup.get("reference_sql")
    reference_sql = reference_sql_raw.strip() if isinstance(reference_sql_raw, str) else None
    meta_raw = lookup.get("meta")
    meta = meta_raw if isinstance(meta_raw, dict) else {}
    return {
        "route": "data",
        "candidate_sql": reference_sql,
        "response_text": response,
        "generation_result": {
            "answer_structured": {
                "sql": reference_sql,
                "explanation": response,
                "instruction_type": "schema_lookup",
                "question": question,
            },
            "meta": meta,
        },
    }


def _tool_sql_validate_explain(state: AgentState, args: dict[str, Any]) -> AgentState:
    sql_arg = args.get("sql")
    sql = nodes._coerce_str(sql_arg).strip() if isinstance(sql_arg, str) else ""
    source = "arg_sql"
    if not sql:
        user_sql = state.get("user_sql")
        if isinstance(user_sql, str) and user_sql.strip():
            sql = user_sql.strip()
            source = "user_sql"
        else:
            last_sql = nodes._coerce_str(state.get("last_candidate_sql")).strip()
            if last_sql:
                sql = last_sql
                source = "last_candidate_sql"
    if not sql:
        message = "검증할 SQL이 없습니다. SQL 코드 블록을 보내거나 먼저 쿼리를 생성해주세요."
        return {
            "route": "data",
            "response_text": message,
            "generation_result": {
                "answer_structured": {
                    "sql": None,
                    "explanation": message,
                    "instruction_type": "sql_validate_explain",
                }
            },
        }

    last_candidate_sql = nodes._coerce_str(state.get("last_candidate_sql")).strip()
    if source == "last_candidate_sql" or (source == "arg_sql" and sql == last_candidate_sql):
        last_dry_run = state.get("last_dry_run")
        if isinstance(last_dry_run, dict) and "success" in last_dry_run:
            dry_run = last_dry_run
        else:
            dry_run = nodes.dry_run_bigquery_sql(sql)
    else:
        dry_run = nodes.dry_run_bigquery_sql(sql)
    normalized_sql = nodes._coerce_str(dry_run.get("sql"), sql).strip() if dry_run else sql
    explanation = (
        "SQL dry-run 검증을 통과했습니다."
        if dry_run.get("success")
        else f"SQL dry-run 검증 실패: {dry_run.get('error', '-')}"
    )
    result: AgentState = {
        "route": "data",
        "candidate_sql": normalized_sql or sql,
        "dry_run": dry_run,
        "response_text": explanation,
        "generation_result": {
            "answer_structured": {
                "sql": normalized_sql or sql,
                "explanation": explanation,
                "instruction_type": "sql_validate_explain",
            }
        },
    }
    if dry_run.get("success"):
        result["last_candidate_sql"] = normalized_sql or sql
        result["last_dry_run"] = dry_run
    return result


def _tool_sql_execute(state: AgentState, args: dict[str, Any]) -> AgentState:
    sql_arg = args.get("sql")
    sql = nodes._coerce_str(sql_arg).strip() if isinstance(sql_arg, str) else ""
    if not sql:
        resolved = _resolve_sql_from_state(state)
        sql = resolved or ""
    if not sql:
        message = "실행할 SQL이 없습니다. SQL 코드 블록을 제공하거나 이전 쿼리를 생성해주세요."
        return {
            "route": "data",
            "candidate_sql": None,
            "execution": {"success": False, "error": message},
            "generation_result": {
                "answer_structured": {
                    "sql": None,
                    "explanation": "실행 요청을 처리할 SQL을 찾지 못했습니다.",
                    "instruction_type": "sql_execute",
                }
            },
        }

    working: AgentState = cast(AgentState, dict(state))
    if sql:
        current_sql = nodes._coerce_str(working.get("candidate_sql")).strip()
        current_dry_run = working.get("dry_run")
        has_usable_dry_run = isinstance(current_dry_run, dict) and "success" in current_dry_run
        pending_sql = nodes._coerce_str(working.get("pending_execution_sql")).strip()
        pending_dry_run = working.get("pending_execution_dry_run")
        has_pending_dry_run = isinstance(pending_dry_run, dict) and "success" in pending_dry_run
        last_sql = nodes._coerce_str(working.get("last_candidate_sql")).strip()
        last_dry_run = working.get("last_dry_run")
        has_last_dry_run = isinstance(last_dry_run, dict) and "success" in last_dry_run
        if current_sql == sql and has_usable_dry_run:
            pass
        elif pending_sql == sql and has_pending_dry_run:
            working["candidate_sql"] = sql
            working["dry_run"] = cast(dict[str, Any], pending_dry_run)
        elif last_sql == sql and has_last_dry_run:
            working["candidate_sql"] = sql
            working["dry_run"] = cast(dict[str, Any], last_dry_run)
        else:
            working["candidate_sql"] = sql
            working["dry_run"] = nodes.dry_run_bigquery_sql(sql)

    guarded = _run_guarded_execute(working, action="sql_execute")
    execution_payload = guarded.get("execution")
    execution = dict(execution_payload) if isinstance(execution_payload, dict) else {}
    policy = nodes._coerce_str(guarded.get("execution_policy")).strip()
    policy_reason = nodes._coerce_str(guarded.get("execution_policy_reason")).strip()

    if policy == "approval_required":
        request_id = _request_id(state)
        execution["status"] = "pending_approval"
        execution["request"] = {
            "id": request_id,
            "sql": guarded.get("candidate_sql"),
            "dry_run": guarded.get("dry_run") or {},
            "reason": policy_reason,
            "estimated_cost_usd": guarded.get("estimated_cost_usd"),
        }
        execution["success"] = False
    elif execution.get("success") is True:
        execution["status"] = "succeeded"
        execution.pop("request", None)
    else:
        execution["status"] = "failed"
        execution.pop("request", None)

    result: AgentState = cast(AgentState, dict(guarded))
    result["route"] = "data"
    result["execution"] = execution
    result["response_text"] = nodes._coerce_str(execution.get("error"), "")
    if isinstance(result.get("candidate_sql"), str) and isinstance(result.get("dry_run"), dict):
        if result["dry_run"].get("success"):
            result["last_candidate_sql"] = result["candidate_sql"]
            result["last_dry_run"] = result["dry_run"]
    return result


def _tool_execution_approve(state: AgentState, args: dict[str, Any]) -> AgentState:
    del args
    guarded = _run_guarded_execute(state, action="execution_approve")
    execution_payload = guarded.get("execution")
    execution = dict(execution_payload) if isinstance(execution_payload, dict) else {}
    if execution.get("success") is True:
        execution["status"] = "succeeded"
    else:
        execution["status"] = "failed"
    execution.pop("request", None)
    result: AgentState = cast(AgentState, dict(guarded))
    result["route"] = "data"
    result["execution"] = execution
    result["response_text"] = nodes._coerce_str(execution.get("error"), "")
    return result


def _tool_execution_cancel(state: AgentState, args: dict[str, Any]) -> AgentState:
    del args
    guarded = _run_guarded_execute(state, action="execution_cancel")
    execution_payload = guarded.get("execution")
    execution = dict(execution_payload) if isinstance(execution_payload, dict) else {}
    execution["status"] = "cancelled"
    execution.pop("request", None)
    result: AgentState = cast(AgentState, dict(guarded))
    result["route"] = "data"
    result["execution"] = execution
    result["response_text"] = nodes._coerce_str(execution.get("error"), "")
    return result


_TOOL_REGISTRY: dict[str, ToolExecutor] = {
    "schema_lookup": _tool_schema_lookup,
    "sql_validate_explain": _tool_sql_validate_explain,
    "sql_execute": _tool_sql_execute,
    "execution_approve": _tool_execution_approve,
    "execution_cancel": _tool_execution_cancel,
}


def _node_prepare(state: AgentState) -> AgentState:
    working: AgentState = cast(AgentState, dict(state))
    working["runtime_mode"] = "loop"

    _apply_patch(working, nodes._node_reset_turn_state(working))
    _apply_patch(working, nodes._node_classify_relevance(working))
    _ensure_turn_state(working)
    working["tool_calls"] = []
    working["agent_should_finalize"] = False
    if not working.get("should_respond"):
        _apply_patch(working, nodes._node_clear_response(working))
        turn = working.get("turn") or {}
        turn["terminated"] = True
        turn["termination_reason"] = "irrelevant"
        working["turn"] = turn
        return cast(AgentState, working)

    _apply_patch(working, nodes._node_ingest(working))
    return cast(AgentState, working)


def _node_agent(state: AgentState) -> AgentState:
    working: AgentState = cast(AgentState, dict(state))
    _ensure_turn_state(working)

    turn = working.get("turn") or {}
    step = int(turn.get("step", 0)) + 1
    max_steps = int(turn.get("max_steps", _get_loop_max_steps()))
    turn["step"] = step
    working["turn"] = turn
    working["tool_calls"] = []
    working["agent_should_finalize"] = False

    if step > max_steps:
        turn["terminated"] = True
        turn["termination_reason"] = "max_steps_exceeded"
        working["turn"] = turn
        if not nodes._coerce_str(working.get("response_text")).strip():
            working["response_text"] = (
                "요청을 처리하는 중 내부 단계 제한에 도달했습니다. "
                "질문을 조금 더 구체적으로 나눠서 다시 요청해주세요."
            )
        working["agent_should_finalize"] = True
        return cast(AgentState, working)

    _apply_patch(working, nodes._node_plan_turn_action(working))
    action = nodes._coerce_turn_action(working.get("action"), default="chat_reply")

    tool_calls: list[ToolCall] = []
    if action == "chat_reply":
        _apply_patch(working, nodes._node_chat_reply(working))
        working["agent_should_finalize"] = True
    elif action == "sql_generate":
        _apply_patch(working, nodes._node_sql_generate(working))
        sql = nodes._coerce_str(working.get("candidate_sql")).strip() or None
        args: dict[str, Any] = {}
        if sql:
            args["sql"] = sql
        tool_calls = [{"id": f"tc-{step}-1", "name": "sql_execute", "args": args}]
    elif action == "schema_lookup":
        tool_calls = [
            {
                "id": f"tc-{step}-1",
                "name": "schema_lookup",
                "args": {"question": working.get("text") or ""},
            }
        ]
    elif action == "sql_validate_explain":
        sql = _resolve_sql_from_state(working)
        args = {"sql": sql} if isinstance(sql, str) and sql.strip() else {}
        tool_calls = [{"id": f"tc-{step}-1", "name": "sql_validate_explain", "args": args}]
    elif action == "sql_execute":
        sql = _resolve_sql_from_state(working)
        args = {"sql": sql} if isinstance(sql, str) and sql.strip() else {}
        tool_calls = [{"id": f"tc-{step}-1", "name": "sql_execute", "args": args}]
    elif action == "execution_approve":
        execution = working.get("execution")
        request_id = None
        if isinstance(execution, dict):
            request = execution.get("request")
            if isinstance(request, dict):
                request_id = request.get("id")
        args = {"request_id": request_id} if isinstance(request_id, str) and request_id else {}
        tool_calls = [{"id": f"tc-{step}-1", "name": "execution_approve", "args": args}]
    elif action == "execution_cancel":
        execution = working.get("execution")
        request_id = None
        if isinstance(execution, dict):
            request = execution.get("request")
            if isinstance(request, dict):
                request_id = request.get("id")
        args = {"request_id": request_id} if isinstance(request_id, str) and request_id else {}
        tool_calls = [{"id": f"tc-{step}-1", "name": "execution_cancel", "args": args}]
    else:
        working["action"] = "chat_reply"
        _apply_patch(working, nodes._node_chat_reply(working))
        working["agent_should_finalize"] = True

    if tool_calls:
        working["tool_calls"] = [dict(call) for call in tool_calls]
    return cast(AgentState, working)


def _node_tools(state: AgentState) -> AgentState:
    working: AgentState = cast(AgentState, dict(state))
    tool_calls = _coerce_tool_calls(working.get("tool_calls"))
    if not tool_calls:
        working["agent_should_finalize"] = True
        return cast(AgentState, working)

    terminate_turn = False
    termination_reason = ""

    for call in tool_calls:
        name = call["name"]
        args = call["args"]
        executor = _TOOL_REGISTRY.get(name)
        if executor is None:
            continue
        updates = executor(working, args)
        _apply_patch(working, updates)

        execution = working.get("execution")
        execution_status = execution.get("status") if isinstance(execution, dict) else None
        if execution_status == "pending_approval":
            terminate_turn = True
            termination_reason = "pending_approval"
        else:
            if execution_status in {"succeeded", "failed", "cancelled"}:
                terminate_turn = True
                termination_reason = f"execution_{execution_status}"
            elif name in {"schema_lookup", "sql_validate_explain"}:
                terminate_turn = True
                termination_reason = f"{name}_completed"

    working["tool_calls"] = []
    if terminate_turn:
        turn = working.get("turn") or {}
        turn["terminated"] = True
        turn["termination_reason"] = termination_reason
        working["turn"] = turn
    working["agent_should_finalize"] = terminate_turn
    return cast(AgentState, working)


def _node_finalize(state: AgentState) -> AgentState:
    working: AgentState = cast(AgentState, dict(state))
    _apply_patch(working, nodes._node_compose_response(working))
    return working


def _route_prepare(state: AgentState) -> str:
    if not state.get("should_respond"):
        return "finalize"
    return "agent"


def _route_agent(state: AgentState) -> str:
    if state.get("agent_should_finalize"):
        return "finalize"
    if _coerce_tool_calls(state.get("tool_calls")):
        return "tools"
    return "finalize"


def _route_tools(state: AgentState) -> str:
    turn = state.get("turn")
    if isinstance(turn, dict) and bool(turn.get("terminated")):
        return "finalize"
    return "agent"


def _build_graph() -> LoopGraphBuilder:
    graph = StateGraph(AgentState)
    graph.add_node("prepare_node", _node_prepare)
    graph.add_node("agent_node", _node_agent)
    graph.add_node("tools_node", _node_tools)
    graph.add_node("finalize_node", _node_finalize)

    graph.add_edge(START, "prepare_node")
    graph.add_conditional_edges(
        "prepare_node",
        _route_prepare,
        {"agent": "agent_node", "finalize": "finalize_node"},
    )
    graph.add_conditional_edges(
        "agent_node",
        _route_agent,
        {"tools": "tools_node", "finalize": "finalize_node"},
    )
    graph.add_conditional_edges(
        "tools_node",
        _route_tools,
        {"agent": "agent_node", "finalize": "finalize_node"},
    )
    graph.add_edge("finalize_node", END)
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
