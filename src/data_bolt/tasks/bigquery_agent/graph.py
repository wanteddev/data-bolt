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
    graph.add_node("classify_intent_llm", nodes._node_classify_intent_llm)
    graph.add_node("free_chat_planner", nodes._node_free_chat_planner)
    graph.add_node("data_generate_or_validate", nodes._node_data_generate_or_validate)
    graph.add_node("validate_candidate_sql", nodes._node_validate_candidate_sql)
    graph.add_node("policy_gate", nodes._node_policy_gate)
    graph.add_node("execute_sql", nodes._node_execute)
    graph.add_node("free_chat_respond", nodes._node_free_chat_respond)
    graph.add_node("compose_response", nodes._node_compose_response)

    graph.add_edge(START, "reset_turn_state")
    graph.add_edge("reset_turn_state", "classify_relevance")
    graph.add_conditional_edges(
        "classify_relevance",
        nodes._route_relevance,
        {"proceed": "ingest", "stop": "clear_response"},
    )
    graph.add_edge("ingest", "classify_intent_llm")
    graph.add_edge("clear_response", END)
    graph.add_conditional_edges(
        "classify_intent_llm",
        nodes._route_intent,
        {"route_data": "data_generate_or_validate", "route_chat": "free_chat_planner"},
    )
    graph.add_conditional_edges(
        "free_chat_planner",
        nodes._route_chat_plan,
        {"needs_data": "data_generate_or_validate", "chat_only": "free_chat_respond"},
    )
    graph.add_edge("data_generate_or_validate", "validate_candidate_sql")
    graph.add_edge("validate_candidate_sql", "policy_gate")
    graph.add_conditional_edges(
        "policy_gate",
        nodes._route_execution,
        {
            "execute": "execute_sql",
            "skip_chat": "free_chat_respond",
            "skip_data": "compose_response",
        },
    )
    graph.add_conditional_edges(
        "execute_sql",
        nodes._route_after_policy,
        {"chat": "free_chat_respond", "data": "compose_response"},
    )
    graph.add_conditional_edges(
        "free_chat_respond",
        nodes._route_after_policy,
        {"chat": "compose_response", "data": "compose_response"},
    )
    graph.add_edge("compose_response", END)
    return graph
