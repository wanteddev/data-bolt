"""Types for BigQuery agent graph orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

TurnAction = Literal[
    "ignore",
    "chat_reply",
    "schema_lookup",
    "sql_validate_explain",
    "sql_generate",
    "sql_execute",
    "execution_approve",
    "execution_cancel",
]
Route = Literal["data", "chat"]
ExecutionPolicy = Literal["auto_execute", "approval_required", "blocked", ""]


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
    action: TurnAction | None
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
    runtime_mode: str
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
    analysis_brief: dict[str, Any]
    should_respond: bool
    intent_mode: str
    execution_intent: str
    needs_clarification: bool
    clarifying_question: str
    action: TurnAction
    user_sql: str | None
    candidate_sql: str | None
    dry_run: dict[str, Any]
    last_candidate_sql: str | None
    last_dry_run: dict[str, Any]
    pending_execution_sql: str | None
    pending_execution_dry_run: dict[str, Any]
    generation_result: dict[str, Any]
    execution: dict[str, Any]
    response_text: str
    error: str | None
    route: Route
    turn_mode: str
    planned_tool: str
    action_confidence: float
    action_reason: str
    fallback_used: bool
    execution_policy: ExecutionPolicy
    execution_policy_reason: str
    cost_threshold_usd: float
    estimated_cost_usd: float | None
    tool_calls: list[dict[str, Any]]
    turn: dict[str, Any]
    agent_should_finalize: bool
