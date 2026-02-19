from types import SimpleNamespace

import pytest
from pydantic_ai import ApprovalRequired, ModelRetry

from data_bolt.tasks.analyst_agent import tools
from data_bolt.tasks.analyst_agent.deps import AnalystDeps, QueryPolicy
from data_bolt.tasks.analyst_agent.models import SchemaContext


class DummyRetriever:
    def __init__(self) -> None:
        self.called = False

    def search(self, question: str, top_k: int = 6) -> SchemaContext:
        self.called = True
        return SchemaContext(raw_table_info=f"schema:{question}", notes=[str(top_k)])


def _build_ctx(*, tool_call_approved: bool = False):
    deps = AnalystDeps(
        bq_client=None,
        schema_retriever=DummyRetriever(),
        policy=QueryPolicy(max_bytes_without_approval=100, max_bytes_hard_limit=200),
    )
    return SimpleNamespace(deps=deps, tool_call_approved=tool_call_approved)


def test_tool_get_schema_context_uses_retriever() -> None:
    ctx = _build_ctx()
    result = tools.tool_get_schema_context(ctx, "가입자")

    assert result.raw_table_info == "schema:가입자"
    assert ctx.deps.last_schema is not None


def test_tool_bigquery_dry_run_raises_model_retry_on_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        tools,
        "dry_run_bigquery_sql",
        lambda _sql: {
            "success": False,
            "error": "syntax error",
            "total_bytes_processed": None,
            "estimated_cost_usd": None,
        },
    )

    ctx = _build_ctx()
    with pytest.raises(ModelRetry):
        tools.tool_bigquery_dry_run(ctx, "SELEC 1")


def test_tool_bigquery_dry_run_auth_error_returns_invalid_result(monkeypatch) -> None:
    monkeypatch.setattr(
        tools,
        "dry_run_bigquery_sql",
        lambda _sql: {
            "success": False,
            "error": "Reauthentication is needed. Please run `gcloud auth application-default login`.",
        },
    )

    ctx = _build_ctx()
    result = tools.tool_bigquery_dry_run(ctx, "SELECT 1")
    assert result.is_valid is False
    assert result.error is not None


def test_tool_bigquery_execute_requires_approval_when_cost_high(monkeypatch) -> None:
    monkeypatch.setattr(
        tools,
        "dry_run_bigquery_sql",
        lambda _sql: {
            "success": True,
            "statement_type": "SELECT",
            "total_bytes_processed": 120,
            "estimated_cost_usd": 1.2,
        },
    )
    monkeypatch.setattr(
        tools,
        "execute_bigquery_sql",
        lambda _sql: {"success": True, "job_id": "job-1", "row_count": 1, "preview_rows": []},
    )

    ctx = _build_ctx(tool_call_approved=False)
    with pytest.raises(ApprovalRequired):
        tools.tool_bigquery_execute(ctx, "SELECT 1")


def test_tool_bigquery_execute_runs_after_approval(monkeypatch) -> None:
    monkeypatch.setattr(
        tools,
        "dry_run_bigquery_sql",
        lambda _sql: {
            "success": True,
            "statement_type": "SELECT",
            "total_bytes_processed": 120,
            "estimated_cost_usd": 1.2,
        },
    )
    monkeypatch.setattr(
        tools,
        "execute_bigquery_sql",
        lambda _sql: {
            "success": True,
            "job_id": "job-approved",
            "row_count": 2,
            "preview_rows": [{"v": 1}],
        },
    )

    ctx = _build_ctx(tool_call_approved=True)
    result = tools.tool_bigquery_execute(ctx, "SELECT 1")

    assert result.success is True
    assert result.job_id == "job-approved"
    assert ctx.deps.last_result is not None


def test_tool_bigquery_execute_returns_failure_summary_on_auth_error(monkeypatch) -> None:
    monkeypatch.setattr(
        tools,
        "dry_run_bigquery_sql",
        lambda _sql: {
            "success": False,
            "error": "Reauthentication is needed. Please run `gcloud auth application-default login`.",
        },
    )

    ctx = _build_ctx(tool_call_approved=True)
    result = tools.tool_bigquery_execute(ctx, "SELECT 1")

    assert result.success is False
    assert "Reauthentication is needed" in (result.error or "")
