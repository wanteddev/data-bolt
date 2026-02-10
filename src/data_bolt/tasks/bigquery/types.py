"""Types for BigQuery task services."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

JsonValue = dict[str, Any] | list[Any]


class SQLBuildPayload(TypedDict, total=False):
    text: NotRequired[str]
    question: NotRequired[str]
    table_info: NotRequired[str]
    glossary_info: NotRequired[str]
    history: NotRequired[list[dict[str, Any]]]
    images: NotRequired[list[dict[str, Any]]]
    instruction_type: NotRequired[str]


class ValidationResult(TypedDict, total=False):
    success: bool
    refined: bool
    attempts: int
    sql: str | None
    error: str | None
    total_bytes_processed: int | None
    estimated_cost_usd: float | None
    cache_hit: bool
    job_id: str | None
    refinement_error: str | None


class ExecutionResult(TypedDict, total=False):
    success: bool
    error: str | None
    job_id: str | None
    row_count: int | None
    preview_rows: list[dict[str, Any]]


class IntentResult(TypedDict, total=False):
    intent: str
    confidence: float
    reason: str
    actions: list[str]


class ChatPlanResult(TypedDict, total=False):
    assistant_response: str
    actions: list[str]
    action_reason: str


class SQLBuildResult(TypedDict, total=False):
    choices: list[dict[str, Any]]
    answer_structured: dict[str, Any]
    raw_response: JsonValue
    validation: ValidationResult
    error: str
    meta: dict[str, Any]
