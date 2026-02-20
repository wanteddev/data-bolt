"""Tool implementations used by the PydanticAI analyst agent."""

from __future__ import annotations

import re
from typing import Any

from pydantic_ai import ApprovalRequired, ModelRetry, RunContext

from data_bolt.tasks.tools import (
    dry_run_bigquery_sql,
    estimate_query_cost_usd,
    execute_bigquery_sql,
)

from .deps import AnalystDeps
from .models import DryRunResult, QueryResultSummary, SchemaContext, TableSnippet

_DML_DDL_RE = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|MERGE|CREATE|DROP|ALTER|TRUNCATE|REPLACE|GRANT|REVOKE|CALL|EXECUTE|BEGIN|COMMIT|ROLLBACK)\b",
    re.IGNORECASE,
)


def _is_dml_ddl(sql: str) -> bool:
    return _DML_DDL_RE.search(sql) is not None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_non_retryable_bq_error(error: str | None) -> bool:
    if not error:
        return False
    lowered = error.lower()
    non_retryable_markers = (
        "reauthentication is needed",
        "application-default login",
        "credential",
        "access denied",
        "permission denied",
        "quota",
        "billing",
    )
    return any(marker in lowered for marker in non_retryable_markers)


def _emit_trace(deps: AnalystDeps, node: str, reason: str) -> None:
    callback = deps.trace_callback
    if callback is None:
        return
    try:
        callback(node, reason)
    except Exception:
        return


def tool_get_schema_context(
    ctx: RunContext[AnalystDeps],
    question: str | None = None,
    top_k: int = 6,
) -> SchemaContext:
    """Retrieve schema context via existing RAG helper."""
    resolved_question = str(question or "").strip() or str(ctx.deps.current_user_text or "").strip()
    if not resolved_question:
        raise ModelRetry(
            "Schema search question is missing. Please provide the metric and time range in one sentence."
        )

    _emit_trace(
        ctx.deps,
        "tool.get_schema_context.call",
        f"스키마 컨텍스트를 조회합니다. question_len={len(resolved_question)}, top_k={top_k}",
    )
    context = ctx.deps.schema_retriever.search(question=resolved_question, top_k=top_k)
    if not context.snippets and context.raw_table_info.strip():
        context.snippets = [
            TableSnippet(table="rag_context", description=context.raw_table_info.strip()[:4000])
        ]
    ctx.deps.last_schema = context
    _emit_trace(
        ctx.deps,
        "tool.get_schema_context.result",
        (
            "스키마 컨텍스트 조회를 완료했습니다. "
            f"snippets={len(context.snippets)}, notes={len(context.notes)}"
        ),
    )
    return context


def tool_bigquery_dry_run(ctx: RunContext[AnalystDeps], sql: str) -> DryRunResult:
    """Validate SQL with BigQuery dry-run using existing task tool."""
    normalized_sql = (sql or "").strip()
    if not normalized_sql:
        raise ModelRetry("SQL is empty. Please provide executable BigQuery SQL.")

    _emit_trace(
        ctx.deps,
        "tool.bigquery_dry_run.call",
        f"BigQuery dry-run을 시작합니다. sql_len={len(normalized_sql)}",
    )
    result = dry_run_bigquery_sql(normalized_sql)
    dry_run = DryRunResult(
        total_bytes_processed=_as_int(result.get("total_bytes_processed")),
        total_bytes_billed=_as_int(result.get("total_bytes_billed")),
        statement_type=(
            str(result.get("statement_type")) if result.get("statement_type") else None
        ),
        is_valid=bool(result.get("success")),
        error=(str(result.get("error")) if result.get("error") else None),
        estimated_cost_usd=_as_float(result.get("estimated_cost_usd")),
        job_id=(str(result.get("job_id")) if result.get("job_id") else None),
        cache_hit=(bool(result.get("cache_hit")) if result.get("cache_hit") is not None else None),
    )
    ctx.deps.last_sql = normalized_sql
    ctx.deps.last_dry_run = dry_run
    _emit_trace(
        ctx.deps,
        "tool.bigquery_dry_run.result",
        (
            "BigQuery dry-run 결과를 받았습니다. "
            f"is_valid={dry_run.is_valid}, bytes={dry_run.total_bytes_processed}, "
            f"error={dry_run.error or '-'}"
        ),
    )

    if not dry_run.is_valid:
        error = dry_run.error or "unknown dry-run failure"
        if _is_non_retryable_bq_error(error):
            return dry_run
        raise ModelRetry(f"BigQuery dry-run failed: {error}")
    return dry_run


def tool_bigquery_execute(ctx: RunContext[AnalystDeps], sql: str) -> QueryResultSummary:
    """Execute SQL with approval-aware policy checks."""
    normalized_sql = (sql or "").strip()
    if not normalized_sql:
        raise ModelRetry("SQL is empty. Please provide SQL to execute.")

    _emit_trace(
        ctx.deps,
        "tool.bigquery_execute.call",
        f"BigQuery execute를 시작합니다. sql_len={len(normalized_sql)}",
    )
    dry_run = tool_bigquery_dry_run(ctx, normalized_sql)
    if not dry_run.is_valid:
        error = dry_run.error or "unknown dry-run failure"
        if _is_non_retryable_bq_error(error):
            summary = QueryResultSummary(
                statement_type=dry_run.statement_type,
                total_bytes_processed=dry_run.total_bytes_processed,
                total_bytes_billed=dry_run.total_bytes_billed,
                estimated_cost_usd=dry_run.estimated_cost_usd,
                actual_cost_usd=(
                    estimate_query_cost_usd(dry_run.total_bytes_billed)
                    if dry_run.total_bytes_billed is not None
                    else None
                ),
                success=False,
                error=error,
            )
            ctx.deps.last_result = summary
            _emit_trace(
                ctx.deps,
                "tool.bigquery_execute.result",
                f"실행을 중단했습니다. dry-run 오류={error}",
            )
            return summary
        raise ModelRetry(f"BigQuery dry-run failed before execute: {error}")

    policy = ctx.deps.policy

    statement_type = (dry_run.statement_type or "").upper()
    is_non_select_statement = bool(statement_type and statement_type != "SELECT")
    if (
        policy.require_approval_for_dml_ddl
        and (_is_dml_ddl(normalized_sql) or is_non_select_statement)
        and not ctx.tool_call_approved
    ):
        _emit_trace(
            ctx.deps,
            "tool.bigquery_execute.approval_required",
            (
                "쓰기/DDL 또는 non-SELECT 쿼리라 승인이 필요합니다. "
                f"statement_type={dry_run.statement_type or '-'}"
            ),
        )
        raise ApprovalRequired(
            metadata={
                "reason": "dml_ddl_or_non_select",
                "statement_type": dry_run.statement_type,
                "estimated_bytes": dry_run.total_bytes_processed,
                "estimated_cost_usd": dry_run.estimated_cost_usd,
            }
        )

    if dry_run.total_bytes_processed is not None:
        if dry_run.total_bytes_processed > policy.max_bytes_hard_limit:
            raise ModelRetry(
                "Estimated bytes exceed hard limit. "
                "Please narrow date range, add partition filters, or aggregate first."
            )

        if (
            dry_run.total_bytes_processed > policy.max_bytes_without_approval
            and not ctx.tool_call_approved
        ):
            _emit_trace(
                ctx.deps,
                "tool.bigquery_execute.approval_required",
                (
                    "예상 스캔 바이트가 임계값을 초과해 승인이 필요합니다. "
                    f"estimated_bytes={dry_run.total_bytes_processed}, "
                    f"threshold={policy.max_bytes_without_approval}"
                ),
            )
            raise ApprovalRequired(
                metadata={
                    "reason": "cost",
                    "estimated_bytes": dry_run.total_bytes_processed,
                    "threshold": policy.max_bytes_without_approval,
                    "estimated_cost_usd": dry_run.estimated_cost_usd,
                }
            )

    execution = execute_bigquery_sql(normalized_sql)
    if not execution.get("success"):
        raise ModelRetry(f"BigQuery execute failed: {execution.get('error') or 'unknown error'}")

    preview_rows_raw = execution.get("preview_rows")
    preview_rows = preview_rows_raw if isinstance(preview_rows_raw, list) else []

    execution_total_bytes_processed = _as_int(execution.get("total_bytes_processed"))
    execution_total_bytes_billed = _as_int(execution.get("total_bytes_billed"))

    summary = QueryResultSummary(
        job_id=(str(execution.get("job_id")) if execution.get("job_id") else None),
        statement_type=(dry_run.statement_type or "SELECT"),
        rows_preview=preview_rows,
        row_count=_as_int(execution.get("row_count")),
        total_bytes_processed=(
            execution_total_bytes_processed
            if execution_total_bytes_processed is not None
            else dry_run.total_bytes_processed
        ),
        total_bytes_billed=(
            execution_total_bytes_billed
            if execution_total_bytes_billed is not None
            else dry_run.total_bytes_billed
        ),
        estimated_cost_usd=dry_run.estimated_cost_usd,
        actual_cost_usd=_as_float(execution.get("actual_cost_usd")),
        success=True,
        error=None,
    )
    ctx.deps.last_result = summary
    _emit_trace(
        ctx.deps,
        "tool.bigquery_execute.result",
        (
            "BigQuery execute를 완료했습니다. "
            f"success={summary.success}, row_count={summary.row_count}, job_id={summary.job_id or '-'}"
        ),
    )
    return summary


def register_analyst_tools(agent: Any) -> None:
    """Register all analyst tools on the given agent instance."""
    agent.tool(tool_get_schema_context, name="get_schema_context")
    agent.tool(tool_bigquery_dry_run, name="bigquery_dry_run")
    agent.tool(tool_bigquery_execute, name="bigquery_execute")
