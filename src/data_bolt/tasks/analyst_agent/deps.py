"""Dependency container for analyst agent execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .models import DryRunResult, QueryResultSummary, SchemaContext


class SchemaRetriever(Protocol):
    """Protocol for schema retrieval."""

    def search(self, question: str, top_k: int = 6) -> SchemaContext:
        """Return schema context for a question."""
        ...


@dataclass
class QueryPolicy:
    """Execution policy thresholds."""

    max_bytes_without_approval: int = 5 * 1024**3
    max_bytes_hard_limit: int = 50 * 1024**3
    preview_rows: int = 50
    require_approval_for_dml_ddl: bool = True


@dataclass
class AnalystDeps:
    """Runtime dependencies and mutable per-turn state."""

    bq_client: Any
    schema_retriever: SchemaRetriever
    policy: QueryPolicy
    default_project: str | None = None
    default_dataset: str | None = None
    location: str | None = None

    requester_user_id: str | None = None
    channel_id: str | None = None
    thread_ts: str | None = None

    last_schema: SchemaContext | None = None
    last_dry_run: DryRunResult | None = None
    last_result: QueryResultSummary | None = None
    last_sql: str | None = None
