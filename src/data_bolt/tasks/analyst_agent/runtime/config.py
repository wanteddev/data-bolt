"""Runtime configuration and dependency builders."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from pydantic_ai import DeferredToolRequests, UsageLimits

from data_bolt.tasks.tools import lookup_schema_rag_context

from ..deps import AnalystDeps, QueryPolicy, SchemaRetriever
from ..models import AnalystReply, AskUser, SchemaContext, TableSnippet

OUTPUT_TYPES = [AskUser, AnalystReply, str, DeferredToolRequests]
DEFAULT_USAGE_LIMITS = UsageLimits(
    tool_calls_limit=16,
    request_limit=10,
)

ANALYST_SYSTEM_INSTRUCTIONS = (
    "You are Data Bolt, a senior data analyst for Wantedlab. "
    "Always speak natural, concise Korean.\n"
    "Business context (internal wiki):\n"
    "- Wanted's core business is hiring/career platform operations.\n"
    "- Major product lines include Matchup talent search, career contents/events (ex. Wanted+), "
    "Wanted Gigs freelancer matching, and HR solution offerings.\n"
    "- Common KPI vocabulary includes signup, DAU/WAU/MAU, apply, document response rate, "
    "final hire, and active company metrics.\n"
    "Data context (BigQuery project `wanted-data`):\n"
    "- Prefer aggregated tables in `wanted_stats` for trend questions first.\n"
    "- `wanted_stats.signup_user(create_date, country, user_cnt)` "
    "for daily signup counts.\n"
    "- `wanted_stats.user_stats(date, country, annual, job_category, user_total, user_active)` "
    "and `wanted_stats.user_stats_YM(...)` for total/active user stats.\n"
    "- `wanted_stats.monthly_company_activity_log_w_status` and "
    "`wanted_stats.weekly_company_activity_log_w_status` for active company status.\n"
    "- `analytics_mart.apply` is the core application table "
    "(apply_type, cancel_time, hire_time, user_id, position_id, company_id).\n"
    "Domain conventions:\n"
    "- `wanted_stats.user_stats` / `wanted_stats.user_stats_YM` contain `user_total` and "
    "`user_active`; do not assume a `new_user` column there.\n"
    "- For 신규 가입자수, use `wanted_stats.signup_user.user_cnt` by default "
    "and aggregate by month when needed.\n"
    "- For monthly 신규 가입자 SQL, prefer this pattern:\n"
    "  SELECT SUM(user_cnt) AS new_signups\n"
    "  FROM wanted_stats.signup_user\n"
    "  WHERE create_date >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH), MONTH)\n"
    "    AND create_date < DATE_TRUNC(CURRENT_DATE(), MONTH)\n"
    "- Do not use placeholder table/column names such as `users`, `signup_date`, `가입일자`.\n"
    "- Unless user says otherwise, interpret `지원` as 일반지원 and exclude canceled records "
    "(apply_type='일반지원' AND cancel_time IS NULL).\n"
    "- Interpret `최종합격` as records where hire_time IS NOT NULL.\n"
    "- AU metrics mean distinct active users within the requested period.\n"
    "- Never invent table or column names. If uncertain, call schema tool or ask a clarification.\n"
    "Execution rules:\n"
    "- If schema context is needed, call `get_schema_context` first.\n"
    "- Before execute, always validate SQL with `bigquery_dry_run`.\n"
    "- Keep SQL read-only unless user explicitly asks for writes.\n"
    "- If information is missing, ask concise clarification questions in one turn.\n"
    "- For common metrics (신규 가입자, 전체/활성 유저수), do not ask user for table names first; "
    "start with known tables and only ask if metric definition is ambiguous.\n"
    "- For expensive or risky execution, explain approval requirement clearly.\n"
    "Output rules:\n"
    "- Prefer AskUser or AnalystReply outputs.\n"
    "- In AnalystReply.message, write plain conversational Korean.\n"
    "- Avoid flattery, excessive politeness, and motivational filler.\n"
    '- Never return wrapper JSON like {"answer": ...}, {"thoughts": ...}, or '
    '{"next_action": ...} as the final user-facing message.\n'
    "- If SQL is included, put SQL in the sql field and keep message focused on intent/result.\n"
)


class DefaultSchemaRetriever(SchemaRetriever):
    """Default schema retriever backed by existing RAG lookup helper."""

    def search(self, question: str, top_k: int = 6) -> SchemaContext:
        del top_k
        raw = lookup_schema_rag_context(question)
        table_info = str(raw.get("table_info") or "")
        glossary_info = str(raw.get("glossary_info") or "")
        raw_meta = raw.get("meta")
        meta = raw_meta if isinstance(raw_meta, dict) else {}

        snippets: list[TableSnippet] = []
        if table_info.strip():
            snippets.append(
                TableSnippet(
                    table="rag_context",
                    description=table_info.strip()[:4000],
                )
            )
        notes = [glossary_info.strip()] if glossary_info.strip() else []
        return SchemaContext(
            snippets=snippets,
            notes=notes,
            raw_table_info=table_info,
            raw_glossary_info=glossary_info,
            meta=meta,
        )


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def build_policy() -> QueryPolicy:
    return QueryPolicy(
        max_bytes_without_approval=env_int(
            "BIGQUERY_AGENT_MAX_BYTES_WITHOUT_APPROVAL", 5 * 1024**3
        ),
        max_bytes_hard_limit=env_int("BIGQUERY_AGENT_MAX_BYTES_HARD_LIMIT", 50 * 1024**3),
        preview_rows=env_int("BIGQUERY_RESULT_PREVIEW_ROWS", 50),
        require_approval_for_dml_ddl=True,
    )


def build_deps(payload: Mapping[str, Any]) -> AnalystDeps:
    return AnalystDeps(
        bq_client=None,
        schema_retriever=DefaultSchemaRetriever(),
        policy=build_policy(),
        default_project=os.getenv("BIGQUERY_PROJECT_ID") or None,
        default_dataset=os.getenv("BIGQUERY_DATASET") or None,
        location=os.getenv("BIGQUERY_LOCATION") or None,
        requester_user_id=str(payload.get("user_id") or ""),
        channel_id=str(payload.get("channel_id") or ""),
        thread_ts=str(payload.get("thread_ts") or ""),
        current_user_text=str(payload.get("text") or payload.get("question") or ""),
    )
