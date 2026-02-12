"""Tool wrappers for external BigQuery/RAG side effects."""

from __future__ import annotations

from typing import Any

from . import execution, rag


class RagContextTool:
    """Resolve table/glossary context for a natural-language question."""

    def run(self, *, question: str, context_hints: dict[str, Any] | None = None) -> dict[str, Any]:
        del context_hints
        return rag._collect_rag_context(question)


class DryRunTool:
    """Execute BigQuery dry-run and return normalized metadata."""

    def run(self, *, sql: str) -> dict[str, Any]:
        return execution.dry_run_bigquery_sql(sql)


class ExecuteQueryTool:
    """Execute a BigQuery query and return result preview metadata."""

    def run(self, *, sql: str) -> dict[str, Any]:
        return execution.execute_bigquery_sql(sql)


rag_context_tool = RagContextTool()
dry_run_tool = DryRunTool()
execute_query_tool = ExecuteQueryTool()
