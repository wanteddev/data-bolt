"""Graph construction for BigQuery agent."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from . import nodes
from .types import AgentState

AgentGraphBuilder = StateGraph[AgentState, None, Any, Any]
CompiledAgentGraph = CompiledStateGraph[AgentState, None, Any, Any]


def _build_graph() -> AgentGraphBuilder:
    graph = StateGraph(AgentState)
    graph.add_node("reset_turn_state", nodes._node_reset_turn_state)
    graph.add_node("clear_response", nodes._node_clear_response)
    graph.add_node("ingest", nodes._node_ingest)
    graph.add_node("classify_relevance", nodes._node_classify_relevance)
    graph.add_node("plan_turn_action", nodes._node_plan_turn_action)
    graph.add_node("chat_reply", nodes._node_chat_reply)
    graph.add_node("schema_lookup", nodes._node_schema_lookup)
    graph.add_node("sql_validate_explain", nodes._node_sql_validate_explain)
    graph.add_node("sql_generate", nodes._node_sql_generate)
    graph.add_node("sql_execute", nodes._node_sql_execute)
    graph.add_node("validate_candidate_sql", nodes._node_validate_candidate_sql)
    graph.add_node("policy_gate", nodes._node_policy_gate)
    graph.add_node("execute_sql", nodes._node_execute)
    graph.add_node("compose_response", nodes._node_compose_response)

    graph.add_edge(START, "reset_turn_state")
    graph.add_edge("reset_turn_state", "classify_relevance")
    graph.add_conditional_edges(
        "classify_relevance",
        nodes._route_relevance,
        {"proceed": "ingest", "stop": "clear_response"},
    )
    graph.add_edge("ingest", "plan_turn_action")
    graph.add_edge("clear_response", END)
    graph.add_conditional_edges(
        "plan_turn_action",
        nodes._route_action_node,
        {
            "chat_reply": "chat_reply",
            "schema_lookup": "schema_lookup",
            "sql_validate_explain": "sql_validate_explain",
            "sql_generate": "sql_generate",
            "sql_execute": "sql_execute",
            "policy_gate": "policy_gate",
        },
    )
    graph.add_edge("chat_reply", "compose_response")
    graph.add_edge("schema_lookup", "compose_response")
    graph.add_edge("sql_validate_explain", "compose_response")
    graph.add_edge("sql_generate", "validate_candidate_sql")
    graph.add_edge("sql_execute", "validate_candidate_sql")
    graph.add_edge("validate_candidate_sql", "policy_gate")
    graph.add_conditional_edges(
        "policy_gate",
        nodes._route_execution,
        {
            "execute": "execute_sql",
            "skip": "compose_response",
        },
    )
    graph.add_edge("execute_sql", "compose_response")
    graph.add_edge("compose_response", END)
    return graph
