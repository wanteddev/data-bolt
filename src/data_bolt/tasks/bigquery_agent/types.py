"""Types for BigQuery agent graph orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

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
