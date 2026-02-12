"""LangGraph workflow for SQL generation/validation/refine loop."""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .types import SQLBuildResult, SQLWorkflowState, ValidationResult

WorkflowGraphBuilder = StateGraph[SQLWorkflowState, None, Any, Any]
CompiledWorkflowGraph = CompiledStateGraph[SQLWorkflowState, None, Any, Any]

_workflow_graph_cache: dict[str, CompiledWorkflowGraph] = {}


def _append_trace(state: SQLWorkflowState, node_name: str) -> list[str]:
    trace = list(state.get("workflow_trace") or [])
    trace.append(node_name)
    return trace


def _empty_validation_meta() -> ValidationResult:
    return {
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


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _node_init_context(state: SQLWorkflowState) -> SQLWorkflowState:
    return {
        "refine_attempts": 0,
        "dry_run_success": False,
        "dry_run_meta": {},
        "current_sql": "",
        "generated_sql": None,
        "explanation": "",
        "error": None,
        "refine_candidate_found": False,
        "continue_refine": False,
        "validation_meta": _empty_validation_meta(),
        "workflow_trace": _append_trace(state, "init_context"),
    }


def _node_generate_sql(state: SQLWorkflowState) -> SQLWorkflowState:
    deps = state.get("dependencies") or {}
    generate_bigquery_response = deps["generate_bigquery_response"]
    adapt_response = deps["adapt_response"]
    normalize_contract = deps["normalize_contract"]
    extract_llm_model = deps["extract_llm_model"]

    instruction_type = state.get("instruction_type", "")
    provider = state.get("provider", "")
    try:
        raw_resp = generate_bigquery_response(
            question=state.get("question", ""),
            table_info=state.get("table_info", ""),
            glossary_info=state.get("glossary_info", ""),
            history=state.get("history") or [],
            images=state.get("images") or [],
            instruction_type=instruction_type,
        )
        response = adapt_response(raw_resp, instruction_type)
        normalized = normalize_contract(
            response, instruction_type=instruction_type, raw_resp=raw_resp
        )
        structured = normalized.get("answer_structured") or {}
        sql_raw = structured.get("sql")
        sql = sql_raw.strip() if isinstance(sql_raw, str) and sql_raw.strip() else None
        explanation_raw = structured.get("explanation")
        explanation = explanation_raw.strip() if isinstance(explanation_raw, str) else ""
        return {
            "raw_response": raw_resp,
            "response": cast(SQLBuildResult, normalized),
            "generated_sql": sql,
            "current_sql": sql or "",
            "explanation": explanation,
            "llm_meta": {
                "provider": provider,
                "called": True,
                "success": True,
                "model": extract_llm_model(raw_resp, provider),
            },
            "workflow_trace": _append_trace(state, "generate_sql"),
        }
    except Exception as e:
        response = normalize_contract(
            {"error": str(e), "meta": {}},
            instruction_type=instruction_type,
            raw_resp=None,
        )
        return {
            "error": str(e),
            "response": cast(SQLBuildResult, response),
            "generated_sql": None,
            "llm_meta": {"provider": provider, "called": True, "success": False, "error": str(e)},
            "workflow_trace": _append_trace(state, "generate_sql"),
        }


def _route_after_generate(state: SQLWorkflowState) -> str:
    return "prepare" if state.get("generated_sql") else "fail"


def _node_prepare_validation_target(state: SQLWorkflowState) -> SQLWorkflowState:
    sql = state.get("current_sql") or state.get("generated_sql") or ""
    return {
        "current_sql": sql,
        "workflow_trace": _append_trace(state, "prepare_validation_target"),
    }


def _node_dry_run_sql(state: SQLWorkflowState) -> SQLWorkflowState:
    deps = state.get("dependencies") or {}
    dry_run_sql = deps["dry_run_sql"]
    ensure_trailing_semicolon = deps["ensure_trailing_semicolon"]
    estimate_query_cost_usd = deps["estimate_query_cost_usd"]

    sql = (state.get("current_sql") or "").strip()
    validation = dict(state.get("validation_meta") or _empty_validation_meta())
    attempts = _coerce_int(validation.get("attempts"), 0) + 1
    validation["attempts"] = attempts
    if not sql:
        validation["error"] = validation.get("error") or "실행할 SQL이 없습니다."
        return {
            "dry_run_success": False,
            "dry_run_meta": {"error": validation["error"]},
            "validation_meta": cast(ValidationResult, validation),
            "workflow_trace": _append_trace(state, "dry_run_sql"),
        }

    ok, meta = dry_run_sql(sql)
    dry_run_meta = meta if isinstance(meta, dict) else {}
    if ok:
        bytes_processed = dry_run_meta.get("total_bytes_processed")
        validation.update(
            {
                "success": True,
                "refined": _coerce_int(state.get("refine_attempts"), 0) > 0,
                "sql": ensure_trailing_semicolon(sql),
                "error": None,
                "total_bytes_processed": bytes_processed,
                "estimated_cost_usd": estimate_query_cost_usd(bytes_processed),
                "cache_hit": bool(dry_run_meta.get("cache_hit")),
                "job_id": dry_run_meta.get("job_id"),
            }
        )
        return {
            "dry_run_success": True,
            "dry_run_meta": dry_run_meta,
            "validation_meta": cast(ValidationResult, validation),
            "workflow_trace": _append_trace(state, "dry_run_sql"),
        }

    validation.update(
        {
            "success": False,
            "error": dry_run_meta.get("error"),
            "total_bytes_processed": dry_run_meta.get("total_bytes_processed"),
            "estimated_cost_usd": None,
            "job_id": dry_run_meta.get("job_id"),
        }
    )
    return {
        "dry_run_success": False,
        "dry_run_meta": dry_run_meta,
        "validation_meta": cast(ValidationResult, validation),
        "workflow_trace": _append_trace(state, "dry_run_sql"),
    }


def _node_decide_refine_or_finish(state: SQLWorkflowState) -> SQLWorkflowState:
    max_refine_attempts = _coerce_int(state.get("max_refine_attempts"), 0)
    refine_attempts = _coerce_int(state.get("refine_attempts"), 0)
    can_refine = (not state.get("dry_run_success")) and refine_attempts < max_refine_attempts
    return {
        "continue_refine": can_refine,
        "workflow_trace": _append_trace(state, "decide_refine_or_finish"),
    }


def _route_after_decide(state: SQLWorkflowState) -> str:
    if state.get("dry_run_success"):
        return "success"
    if state.get("continue_refine"):
        return "refine"
    return "failure"


def _node_refine_sql(state: SQLWorkflowState) -> SQLWorkflowState:
    deps = state.get("dependencies") or {}
    refine_bigquery_sql = deps["refine_bigquery_sql"]
    extract_sql_blocks = deps["extract_sql_blocks"]

    validation = dict(state.get("validation_meta") or _empty_validation_meta())
    refine_attempts = _coerce_int(state.get("refine_attempts"), 0) + 1
    prev_sql = state.get("generated_sql") or state.get("current_sql") or ""
    error = str(validation.get("error") or "")

    refined_text = refine_bigquery_sql(
        question=state.get("question", ""),
        ddl_context=state.get("table_info", ""),
        prev_sql=prev_sql,
        error=error,
    )
    refined_blocks = extract_sql_blocks(refined_text or "")
    if refined_blocks:
        return {
            "refine_attempts": refine_attempts,
            "current_sql": refined_blocks[0].strip(),
            "refine_candidate_found": True,
            "workflow_trace": _append_trace(state, "refine_sql"),
        }

    validation["refinement_error"] = "Refine produced no SQL blocks"
    max_refine_attempts = _coerce_int(state.get("max_refine_attempts"), 0)
    return {
        "refine_attempts": refine_attempts,
        "refine_candidate_found": False,
        "continue_refine": refine_attempts < max_refine_attempts,
        "validation_meta": cast(ValidationResult, validation),
        "workflow_trace": _append_trace(state, "refine_sql"),
    }


def _route_after_refine(state: SQLWorkflowState) -> str:
    if state.get("refine_candidate_found"):
        return "prepare"
    if state.get("continue_refine"):
        return "refine"
    return "failure"


def _node_finalize_success(state: SQLWorkflowState) -> SQLWorkflowState:
    return {
        "workflow_trace": _append_trace(state, "finalize_success"),
    }


def _node_finalize_failure(state: SQLWorkflowState) -> SQLWorkflowState:
    return {
        "workflow_trace": _append_trace(state, "finalize_failure"),
    }


def _build_workflow_graph() -> WorkflowGraphBuilder:
    graph = StateGraph(SQLWorkflowState)
    graph.add_node("init_context", _node_init_context)
    graph.add_node("generate_sql", _node_generate_sql)
    graph.add_node("prepare_validation_target", _node_prepare_validation_target)
    graph.add_node("dry_run_sql", _node_dry_run_sql)
    graph.add_node("decide_refine_or_finish", _node_decide_refine_or_finish)
    graph.add_node("refine_sql", _node_refine_sql)
    graph.add_node("finalize_success", _node_finalize_success)
    graph.add_node("finalize_failure", _node_finalize_failure)

    graph.add_edge(START, "init_context")
    graph.add_edge("init_context", "generate_sql")
    graph.add_conditional_edges(
        "generate_sql",
        _route_after_generate,
        {"prepare": "prepare_validation_target", "fail": "finalize_failure"},
    )
    graph.add_edge("prepare_validation_target", "dry_run_sql")
    graph.add_edge("dry_run_sql", "decide_refine_or_finish")
    graph.add_conditional_edges(
        "decide_refine_or_finish",
        _route_after_decide,
        {"success": "finalize_success", "refine": "refine_sql", "failure": "finalize_failure"},
    )
    graph.add_conditional_edges(
        "refine_sql",
        _route_after_refine,
        {
            "prepare": "prepare_validation_target",
            "refine": "refine_sql",
            "failure": "finalize_failure",
        },
    )
    graph.add_edge("finalize_success", END)
    graph.add_edge("finalize_failure", END)
    return graph


def run_sql_generation_workflow(input_state: SQLWorkflowState) -> SQLWorkflowState:
    graph = _workflow_graph_cache.get("graph")
    if graph is None:
        graph = _build_workflow_graph().compile()
        _workflow_graph_cache["graph"] = graph
    output = cast(Any, graph).invoke(input_state)
    return cast(SQLWorkflowState, output)
