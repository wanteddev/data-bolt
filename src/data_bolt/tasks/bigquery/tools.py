"""Tool wrappers for external BigQuery/RAG side effects."""

from __future__ import annotations

import os
import re
from typing import Any

from . import execution, rag

_BLOCKED_SQL_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE|REPLACE|GRANT|REVOKE|CALL|EXECUTE|BEGIN|COMMIT|ROLLBACK)\b",
    re.IGNORECASE,
)

_dry_run_callable = execution.dry_run_bigquery_sql
_execute_callable = execution.execute_bigquery_sql


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_read_only_sql(sql: str) -> tuple[bool, str]:
    without_block_comments = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    normalized = re.sub(r"--[^\n]*", " ", without_block_comments).strip()
    statements = [part.strip() for part in normalized.split(";") if part.strip()]
    if not statements:
        return False, "실행할 SQL이 없습니다."
    if len(statements) != 1:
        return False, "다중 statement SQL은 실행할 수 없습니다."

    statement = statements[0]
    first_token = statement.split(None, 1)[0].upper() if statement.split() else ""
    if first_token not in {"SELECT", "WITH"}:
        return False, "읽기 전용 SELECT 쿼리만 실행할 수 있습니다."
    if _BLOCKED_SQL_KEYWORDS.search(statement):
        return False, "쓰기/DDL SQL은 실행할 수 없습니다."
    return True, ""


def _get_auto_execute_max_cost_usd() -> float:
    raw = os.getenv("BIGQUERY_AUTO_EXECUTE_MAX_COST_USD", "1.0").strip()
    try:
        value = float(raw)
    except ValueError:
        return 1.0
    return value if value >= 0 else 1.0


class RagContextTool:
    """Resolve table/glossary context for a natural-language question."""

    def run(self, *, question: str, context_hints: dict[str, Any] | None = None) -> dict[str, Any]:
        del context_hints
        return rag._collect_rag_context(question)


class DryRunTool:
    """Execute BigQuery dry-run and return normalized metadata."""

    def run(self, *, sql: str) -> dict[str, Any]:
        return _dry_run_callable(sql)


class ExecuteQueryTool:
    """Execute a BigQuery query and return result preview metadata."""

    def run(self, *, sql: str) -> dict[str, Any]:
        return _execute_callable(sql)


class GuardedExecuteTool:
    """Run strict policy checks and execute SQL through a single guarded chokepoint."""

    def run(
        self,
        *,
        action: str,
        candidate_sql: str | None,
        dry_run: dict[str, Any] | None,
        pending_execution_sql: str | None,
        pending_execution_dry_run: dict[str, Any] | None,
    ) -> dict[str, Any]:
        threshold = _get_auto_execute_max_cost_usd()
        approval_confirmed = action == "execution_approve"
        cancelled = action == "execution_cancel"

        if cancelled:
            return {
                "can_execute": False,
                "execution": {"success": False, "error": "쿼리 실행 요청을 취소했습니다."},
                "pending_execution_sql": None,
                "pending_execution_dry_run": {},
                "execution_policy": "blocked",
                "execution_policy_reason": "execution_cancelled",
                "cost_threshold_usd": threshold,
                "estimated_cost_usd": None,
            }

        sql = (candidate_sql or "").strip()
        policy_dry_run = dry_run if isinstance(dry_run, dict) else {}

        if approval_confirmed:
            pending_sql = (pending_execution_sql or "").strip()
            if not pending_sql:
                return {
                    "can_execute": False,
                    "execution": {
                        "success": False,
                        "error": "승인할 대기 중인 쿼리가 없습니다. 먼저 실행 요청을 보내주세요.",
                    },
                    "pending_execution_sql": None,
                    "pending_execution_dry_run": {},
                    "execution_policy": "blocked",
                    "execution_policy_reason": "approval_without_pending_sql",
                    "cost_threshold_usd": threshold,
                    "estimated_cost_usd": None,
                }
            sql = pending_sql
            pending_dry = (
                pending_execution_dry_run if isinstance(pending_execution_dry_run, dict) else {}
            )
            if not policy_dry_run and pending_dry:
                policy_dry_run = pending_dry

        if not sql:
            return {
                "can_execute": False,
                "execution": {"success": False, "error": "실행할 SQL이 없습니다."},
                "pending_execution_sql": None,
                "pending_execution_dry_run": {},
                "execution_policy": "blocked",
                "execution_policy_reason": "missing_sql",
                "cost_threshold_usd": threshold,
                "estimated_cost_usd": None,
            }

        read_only, reason = _is_read_only_sql(sql)
        if not read_only:
            return {
                "can_execute": False,
                "execution": {"success": False, "error": reason},
                "pending_execution_sql": None,
                "pending_execution_dry_run": {},
                "execution_policy": "blocked",
                "execution_policy_reason": "not_read_only",
                "cost_threshold_usd": threshold,
                "estimated_cost_usd": None,
            }

        if not policy_dry_run:
            policy_dry_run = _dry_run_callable(sql)
        if policy_dry_run.get("success"):
            normalized_sql = policy_dry_run.get("sql")
            if isinstance(normalized_sql, str) and normalized_sql.strip():
                sql = normalized_sql.strip()

        if not policy_dry_run:
            return {
                "can_execute": False,
                "execution": {
                    "success": False,
                    "error": "dry-run 정보가 없어 실행을 보류했습니다.",
                },
                "pending_execution_sql": None,
                "pending_execution_dry_run": {},
                "execution_policy": "blocked",
                "execution_policy_reason": "missing_dry_run",
                "cost_threshold_usd": threshold,
                "estimated_cost_usd": None,
            }

        if not policy_dry_run.get("success", True):
            return {
                "can_execute": False,
                "execution": {"success": False, "error": "dry-run 실패로 실행을 중단했습니다."},
                "pending_execution_sql": None,
                "pending_execution_dry_run": {},
                "execution_policy": "blocked",
                "execution_policy_reason": "dry_run_failed",
                "cost_threshold_usd": threshold,
                "estimated_cost_usd": _coerce_float(policy_dry_run.get("estimated_cost_usd")),
                "dry_run": policy_dry_run,
                "candidate_sql": sql,
            }

        max_bytes = int(os.getenv("BIGQUERY_MAX_BYTES_BILLED", "0"))
        bytes_processed = policy_dry_run.get("total_bytes_processed")
        try:
            bytes_num = int(bytes_processed) if bytes_processed is not None else 0
        except (TypeError, ValueError):
            bytes_num = 0

        if max_bytes > 0 and bytes_num > max_bytes:
            return {
                "can_execute": False,
                "execution": {
                    "success": False,
                    "error": f"예상 처리 바이트가 제한을 초과했습니다. bytes={bytes_num}, limit={max_bytes}",
                },
                "pending_execution_sql": None,
                "pending_execution_dry_run": {},
                "execution_policy": "blocked",
                "execution_policy_reason": "max_bytes_exceeded",
                "cost_threshold_usd": threshold,
                "estimated_cost_usd": _coerce_float(policy_dry_run.get("estimated_cost_usd")),
                "dry_run": policy_dry_run,
                "candidate_sql": sql,
            }

        estimated_cost = _coerce_float(policy_dry_run.get("estimated_cost_usd"))

        if not approval_confirmed and (estimated_cost is None or estimated_cost >= threshold):
            if estimated_cost is None:
                approval_message = (
                    "예상 비용을 계산할 수 없어 실행 전 사용자 승인이 필요합니다. "
                    "위 쿼리와 dry-run 결과를 확인한 뒤 `실행 승인` 또는 `실행 취소`로 답변해주세요."
                )
                policy_reason = "estimated_cost_missing"
            else:
                approval_message = (
                    "예상 비용이 자동 실행 임계값 이상입니다. "
                    f"(estimated={estimated_cost}, threshold={threshold}) "
                    "위 쿼리와 dry-run 결과를 확인한 뒤 `실행 승인` 또는 `실행 취소`로 답변해주세요."
                )
                policy_reason = "cost_above_threshold"
            return {
                "can_execute": False,
                "candidate_sql": sql,
                "dry_run": policy_dry_run,
                "execution": {"success": False, "error": approval_message},
                "pending_execution_sql": sql,
                "pending_execution_dry_run": policy_dry_run,
                "execution_policy": "approval_required",
                "execution_policy_reason": policy_reason,
                "cost_threshold_usd": threshold,
                "estimated_cost_usd": estimated_cost,
            }

        execution_result = _execute_callable(sql)
        reason = "user_approved" if approval_confirmed else "cost_below_threshold"
        return {
            "can_execute": True,
            "candidate_sql": sql,
            "dry_run": policy_dry_run,
            "execution": execution_result,
            "pending_execution_sql": None,
            "pending_execution_dry_run": {},
            "execution_policy": "auto_execute",
            "execution_policy_reason": reason,
            "cost_threshold_usd": threshold,
            "estimated_cost_usd": estimated_cost,
        }


rag_context_tool = RagContextTool()
dry_run_tool = DryRunTool()
execute_query_tool = ExecuteQueryTool()
guarded_execute_tool = GuardedExecuteTool()
