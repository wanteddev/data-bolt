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


class TurnActionResult(TypedDict, total=False):
    action: str
    confidence: float
    reason: str


class ChatPlanResult(TypedDict, total=False):
    assistant_response: str


class ExecutionInsightResult(TypedDict, total=False):
    summary: str
    insight: str
    follow_up_questions: list[str]


class SchemaLookupResult(TypedDict, total=False):
    response_text: str
    reference_sql: str | None
    meta: dict[str, Any]


class SQLValidationExplainResult(TypedDict, total=False):
    response_text: str
    meta: dict[str, Any]


class SQLBuildResult(TypedDict, total=False):
    choices: list[dict[str, Any]]
    answer_structured: dict[str, Any]
    raw_response: JsonValue
    validation: ValidationResult
    error: str
    meta: dict[str, Any]


class SQLWorkflowState(TypedDict, total=False):
    question: str
    table_info: str
    glossary_info: str
    history: list[dict[str, Any]]
    images: list[dict[str, Any]]
    instruction_type: str
    max_refine_attempts: int
    refine_attempts: int
    current_sql: str
    generated_sql: str | None
    explanation: str
    raw_response: JsonValue
    response: SQLBuildResult
    dry_run_success: bool
    dry_run_meta: dict[str, Any]
    validation_meta: ValidationResult
    error: str | None
    provider: str
    llm_meta: dict[str, Any]
    workflow_trace: list[str]
    refine_candidate_found: bool
    continue_refine: bool
    dependencies: dict[str, Any]
