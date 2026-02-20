"""Framework-agnostic BigQuery dry-run and execution helpers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from google.api_core.exceptions import BadRequest, GoogleAPICallError

if TYPE_CHECKING:
    from google.cloud.bigquery import Client as BigQueryClient

BIGQUERY_ON_DEMAND_USD_PER_TB = float(os.getenv("BIGQUERY_ON_DEMAND_USD_PER_TB", "5"))

_bigquery_client: BigQueryClient | None = None


def _ensure_trailing_semicolon(sql: str | None) -> str | None:
    if sql is None:
        return None
    stripped = sql.rstrip()
    return stripped if stripped.endswith(";") else f"{stripped};"


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def estimate_query_cost_usd(
    total_bytes_processed: int | str | None,
    *,
    price_per_tb_usd: float = BIGQUERY_ON_DEMAND_USD_PER_TB,
) -> float | None:
    bytes_processed = _coerce_int(total_bytes_processed)
    if bytes_processed is None or bytes_processed < 0:
        return None
    tebibyte = float(1024**4)
    return round((bytes_processed / tebibyte) * price_per_tb_usd, 6)


def _get_bigquery_client() -> BigQueryClient:
    global _bigquery_client
    if _bigquery_client is None:
        from google.cloud import bigquery

        project = os.getenv("BIGQUERY_PROJECT_ID") or None
        location = os.getenv("BIGQUERY_LOCATION") or None
        if location:
            _bigquery_client = bigquery.Client(project=project, location=location)
        else:
            _bigquery_client = bigquery.Client(project=project)
    return _bigquery_client


def _get_bigquery_location() -> str | None:
    location = os.getenv("BIGQUERY_LOCATION")
    if location and location.strip():
        return location.strip()
    return None


def _get_query_timeout_seconds() -> float:
    raw = os.getenv("BIGQUERY_QUERY_TIMEOUT_SECONDS", "120")
    try:
        timeout = float(raw)
    except ValueError:
        timeout = 120.0
    return timeout if timeout > 0 else 120.0


def _get_max_bytes_billed() -> int | None:
    raw = os.getenv("BIGQUERY_MAX_BYTES_BILLED", "0").strip()
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _format_bigquery_error(error: Exception) -> str:
    if isinstance(error, BadRequest):
        errors = getattr(error, "errors", None)
        if isinstance(errors, list) and errors:
            first = errors[0] if isinstance(errors[0], dict) else {}
            message = first.get("message")
            if isinstance(message, str) and message.strip():
                return message
        if str(error).strip():
            return str(error)
        return "BigQuery dry-run failed with bad request."
    if isinstance(error, GoogleAPICallError) and str(error).strip():
        return str(error)
    return str(error) if str(error).strip() else "Unknown BigQuery error"


def _dry_run_sql(sql: str) -> tuple[bool, dict[str, Any]]:
    if not sql or not sql.strip():
        return False, {"error": "SQL is empty"}

    try:
        from google.cloud import bigquery

        client = _get_bigquery_client()
        config = bigquery.QueryJobConfig(
            dry_run=True,
            use_query_cache=False,
            use_legacy_sql=False,
        )
        max_bytes = _get_max_bytes_billed()
        if max_bytes is not None:
            config.maximum_bytes_billed = max_bytes

        query_job = client.query(sql, job_config=config, location=_get_bigquery_location())
        if query_job.errors:
            first_error = query_job.errors[0] if query_job.errors else {}
            error_text = (
                str(first_error.get("message"))
                if isinstance(first_error, dict) and first_error.get("message")
                else "BigQuery dry-run returned errors"
            )
            return False, {"error": error_text}

        return True, {
            "total_bytes_processed": query_job.total_bytes_processed,
            "job_id": query_job.job_id,
            "cache_hit": query_job.cache_hit,
            "error": None,
        }
    except Exception as exc:
        return False, {"error": _format_bigquery_error(exc)}


def dry_run_bigquery_sql(sql: str) -> dict[str, Any]:
    ok, meta = _dry_run_sql(sql)
    return {
        "success": ok,
        "sql": _ensure_trailing_semicolon(sql) if ok else None,
        "error": meta.get("error"),
        "total_bytes_processed": meta.get("total_bytes_processed"),
        "estimated_cost_usd": estimate_query_cost_usd(meta.get("total_bytes_processed")),
        "job_id": meta.get("job_id"),
        "cache_hit": meta.get("cache_hit"),
    }


def execute_bigquery_sql(sql: str) -> dict[str, Any]:
    try:
        from google.cloud import bigquery

        client = _get_bigquery_client()
        config = bigquery.QueryJobConfig(use_legacy_sql=False)
        max_bytes = _get_max_bytes_billed()
        if max_bytes is not None:
            config.maximum_bytes_billed = max_bytes

        query_job = client.query(sql, job_config=config, location=_get_bigquery_location())
        iterator = query_job.result(timeout=_get_query_timeout_seconds())
        preview_limit = int(os.getenv("BIGQUERY_RESULT_PREVIEW_ROWS", "20"))
        preview_rows: list[dict[str, Any]] = []
        for index, row in enumerate(iterator):
            if index < preview_limit:
                preview_rows.append(dict(row.items()))
            else:
                break

        row_count = iterator.total_rows
        if row_count is None:
            row_count = query_job.num_dml_affected_rows

        return {
            "success": True,
            "error": None,
            "job_id": query_job.job_id,
            "row_count": row_count,
            "preview_rows": preview_rows,
            "total_bytes_processed": query_job.total_bytes_processed,
            "total_bytes_billed": query_job.total_bytes_billed,
            "actual_cost_usd": estimate_query_cost_usd(query_job.total_bytes_billed),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": _format_bigquery_error(exc),
            "job_id": None,
            "row_count": None,
            "preview_rows": [],
        }
