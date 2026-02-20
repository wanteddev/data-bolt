"""Domain models for the PydanticAI-based analyst agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    """Column-level schema hint."""

    name: str
    type: str = ""
    description: str = ""


class TableSnippet(BaseModel):
    """Table-level schema hint."""

    table: str
    description: str = ""
    columns: list[ColumnInfo] = Field(default_factory=list)
    join_hints: list[str] = Field(default_factory=list)


class SchemaContext(BaseModel):
    """Schema and glossary context gathered via RAG."""

    snippets: list[TableSnippet] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    raw_table_info: str = ""
    raw_glossary_info: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class DryRunResult(BaseModel):
    """BigQuery dry-run summary."""

    total_bytes_processed: int | None = None
    total_bytes_billed: int | None = None
    statement_type: str | None = None
    is_valid: bool = True
    error: str | None = None
    estimated_cost_usd: float | None = None
    job_id: str | None = None
    cache_hit: bool | None = None


class QueryResultSummary(BaseModel):
    """BigQuery execution summary."""

    job_id: str | None = None
    statement_type: str | None = None
    rows_preview: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int | None = None
    total_bytes_processed: int | None = None
    total_bytes_billed: int | None = None
    estimated_cost_usd: float | None = None
    actual_cost_usd: float | None = None
    success: bool = True
    error: str | None = None


class AskUser(BaseModel):
    """A clarification request to user."""

    kind: Literal["ask_user"] = "ask_user"
    message: str
    needed: list[str] = Field(default_factory=list)


class AnalystReply(BaseModel):
    """Final analyst response payload."""

    kind: Literal["reply"] = "reply"
    message: str
    sql: str | None = None
    schema_used: SchemaContext | None = None
    dry_run: DryRunResult | None = None
    result: QueryResultSummary | None = None
    next_suggestions: list[str] = Field(default_factory=list)
