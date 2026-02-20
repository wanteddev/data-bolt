"""Background task processor for Slack bot."""

import json
import logging
import os
import traceback
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, ParamSpec, TypeVar

import httpx
from anyio import to_thread
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)

slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
THREAD_HISTORY_LIMIT = int(os.environ.get("SLACK_THREAD_HISTORY_LIMIT", "12"))


P = ParamSpec("P")
T = TypeVar("T")


async def _run_slack_call(func: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
    return await to_thread.run_sync(partial(func, *args, **kwargs))


def _build_thread_history(
    messages: list[dict[str, Any]], message_ts: str | None
) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in messages:
        if message.get("subtype"):
            continue
        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        if message_ts and message.get("ts") == message_ts:
            continue

        role = "assistant" if message.get("bot_id") else "user"
        history.append({"role": role, "content": text.strip()})
    return history


async def _fetch_thread_history(
    channel_id: str | None, thread_ts: str | None, message_ts: str | None
) -> list[dict[str, str]]:
    if not channel_id or not thread_ts:
        return []
    try:
        response = await _run_slack_call(
            slack_client.conversations_replies,
            channel=channel_id,
            ts=thread_ts,
            limit=THREAD_HISTORY_LIMIT,
        )
    except Exception as e:
        logger.warning(f"Failed to fetch thread history: {e}")
        return []

    messages = response.get("messages") if isinstance(response, dict) else None
    if not isinstance(messages, list):
        return []
    return _build_thread_history(messages, message_ts)


async def send_error_response(
    response_url: str | None, channel_id: str | None, error_message: str
) -> None:
    """
    Send error message to user via response_url or Slack API.

    Args:
        response_url: Slack response URL (for slash commands)
        channel_id: Channel ID to send message to (fallback)
        error_message: User-friendly error message
    """
    try:
        if response_url:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    response_url,
                    json={
                        "response_type": "ephemeral",
                        "text": f":x: {error_message}",
                    },
                )
        elif channel_id:
            await _run_slack_call(
                slack_client.chat_postMessage,
                channel=channel_id,
                text=f":x: {error_message}",
            )
    except Exception as e:
        logger.error(f"Failed to send error response: {e}")


async def process_background_task(event: dict[str, Any]) -> dict[str, Any]:
    """
    Process background tasks invoked from the main Slack handler.

    This function runs in a separate Lambda with longer timeout,
    allowing for complex processing without hitting Slack's 3-second limit.

    Args:
        event: Contains 'task_type' and 'payload' from the invoking Lambda

    Returns:
        Result of the background processing
    """
    task_type = event.get("task_type", "unknown")
    payload = event.get("payload", {})

    logger.info(f"Processing background task: {task_type}")

    handlers = {
        "bigquery_sql": handle_bigquery_sql_bg,
        "build_bigquery": handle_bigquery_sql_bg,
        "bigquery_approval": handle_bigquery_approval_bg,
    }

    handler = handlers.get(task_type)
    if not handler:
        logger.warning(f"Unknown task type: {task_type}")
        return {"status": "error", "message": f"Unknown task type: {task_type}"}

    try:
        return await handler(payload)
    except SlackApiError as e:
        error_msg = f"Slack API error: {e.response.get('error', 'unknown')}"
        logger.error(f"Background task {task_type} failed: {error_msg}")

        # Notify user of error
        await send_error_response(
            response_url=payload.get("response_url"),
            channel_id=payload.get("channel_id"),
            error_message="There was an error communicating with Slack. Please try again.",
        )
        return {"status": "error", "message": error_msg}

    except httpx.HTTPError as e:
        error_msg = f"HTTP error: {e}"
        logger.error(f"Background task {task_type} failed: {error_msg}")

        await send_error_response(
            response_url=None,  # Can't use response_url if HTTP failed
            channel_id=payload.get("channel_id"),
            error_message="There was a network error. Please try again.",
        )
        return {"status": "error", "message": error_msg}

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.error(f"Background task {task_type} failed: {error_msg}\n{traceback.format_exc()}")

        await send_error_response(
            response_url=payload.get("response_url"),
            channel_id=payload.get("channel_id"),
            error_message="An unexpected error occurred. Please try again later.",
        )
        return {"status": "error", "message": error_msg}


def _extract_sql_from_result(result: Mapping[str, Any]) -> str | None:
    sql = result.get("candidate_sql")
    if isinstance(sql, str) and sql.strip():
        return sql.strip()

    generation_result = result.get("generation_result")
    if not isinstance(generation_result, dict):
        generation_result = result

    structured = generation_result.get("answer_structured") or {}
    sql = structured.get("sql")
    if isinstance(sql, str) and sql.strip():
        return sql.strip()

    choices = generation_result.get("choices") or []
    if choices:
        message = (choices[0] or {}).get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def _format_validation_summary(validation: dict[str, Any]) -> str | None:
    is_valid = validation.get("is_valid")
    if not isinstance(is_valid, bool):
        is_valid = (
            bool(validation.get("success")) if validation.get("success") is not None else None
        )

    if is_valid is True:
        metrics: list[str] = []
        if validation.get("total_bytes_processed") is not None:
            metrics.append(f"bytes_processed={validation.get('total_bytes_processed')}")
        if validation.get("total_bytes_billed") is not None:
            metrics.append(f"bytes_billed={validation.get('total_bytes_billed')}")
        if validation.get("estimated_cost_usd") is not None:
            metrics.append(f"est_cost_usd={validation.get('estimated_cost_usd')}")
        details = f" ({', '.join(metrics)})" if metrics else ""
        if validation.get("refined"):
            return f":white_check_mark: Dry-run passed after refine.{details}"
        return f":white_check_mark: Dry-run passed.{details}"
    if validation.get("error"):
        return f":warning: Dry-run failed: {validation.get('error')}"
    return None


def _format_execution_summary(execution: dict[str, Any]) -> str | None:
    is_success = execution.get("success")
    if isinstance(is_success, bool) and is_success:
        preview = execution.get("rows_preview")
        if not isinstance(preview, list):
            preview = (
                execution.get("preview_rows")
                if isinstance(execution.get("preview_rows"), list)
                else []
            )
        preview_text = json.dumps(preview, ensure_ascii=False, default=str)[:1200]
        metrics: list[str] = []
        if execution.get("total_bytes_processed") is not None:
            metrics.append(f"bytes_processed={execution.get('total_bytes_processed')}")
        if execution.get("total_bytes_billed") is not None:
            metrics.append(f"bytes_billed={execution.get('total_bytes_billed')}")
        if execution.get("estimated_cost_usd") is not None:
            metrics.append(f"est_cost_usd={execution.get('estimated_cost_usd')}")
        if execution.get("actual_cost_usd") is not None:
            metrics.append(f"actual_cost_usd={execution.get('actual_cost_usd')}")
        metric_line = f"\n{', '.join(metrics)}" if metrics else ""
        return (
            ":rocket: Query executed."
            f"\njob_id={execution.get('job_id')}, row_count={execution.get('row_count')}"
            f"{metric_line}\npreview={preview_text}"
        )

    if execution.get("error"):
        return f":warning: Query execution skipped: {execution.get('error')}"
    return None


def _format_bigquery_response(result: Mapping[str, Any]) -> str:
    parts: list[str] = []
    response_text = result.get("response_text")
    if isinstance(response_text, str) and response_text.strip():
        parts.append(response_text.strip())

    if result.get("error"):
        parts.append(f":x: {result['error']}")

    generation_result = result.get("generation_result")
    if not isinstance(generation_result, dict):
        generation_result = result

    sql = _extract_sql_from_result(result)
    explanation = (generation_result.get("answer_structured") or {}).get("explanation")
    if not parts:
        if sql:
            parts.append(f"```sql\n{sql}\n```")
        if isinstance(explanation, str) and explanation.strip():
            parts.append(explanation.strip())

    validation_raw = result.get("validation")
    validation_data = validation_raw if isinstance(validation_raw, dict) else {}
    validation = _format_validation_summary(validation_data)
    if validation:
        parts.append(validation)

    execution_raw = result.get("execution")
    execution = execution_raw if isinstance(execution_raw, dict) else {}
    execution_summary = _format_execution_summary(execution)
    if execution_summary:
        parts.append(execution_summary)

    if not parts:
        parts.append("No SQL was generated.")
    return "\n\n".join(parts)


async def handle_bigquery_sql_bg(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Background handler for BigQuery SQL generation.

    Expects payload to contain:
    - text/question
    - response_url or channel_id
    - optional thread_ts
    """
    response_url = payload.get("response_url")
    channel_id = payload.get("channel_id")
    thread_ts = payload.get("thread_ts")
    message_ts = payload.get("message_ts")

    logger.info("Background processing BigQuery SQL request")

    include_history = payload.get("include_thread_history", True)
    if include_history and not payload.get("history"):
        should_fetch_history = True
        backend = (os.getenv("BIGQUERY_THREAD_MEMORY_BACKEND") or "memory").strip().lower()
        if backend == "dynamodb":
            try:
                from data_bolt.tasks.analyst_agent import has_thread_memory

                has_memory = await to_thread.run_sync(has_thread_memory, payload)
                should_fetch_history = not has_memory
            except Exception as e:
                logger.warning(f"Failed to check thread memory state: {e}")
                should_fetch_history = True

        if should_fetch_history:
            history = await _fetch_thread_history(channel_id, thread_ts, message_ts)
            if history:
                payload = {**payload, "history": history}

    from data_bolt.tasks.analyst_agent import run_analyst_turn

    result = await to_thread.run_sync(run_analyst_turn, payload)
    should_send_message = bool(result.get("should_respond")) or bool(response_url)
    message = _format_bigquery_response(result) if should_send_message else ""
    message_blocks = (
        result.get("response_blocks") if isinstance(result.get("response_blocks"), list) else None
    )

    try:
        if response_url and should_send_message:
            async with httpx.AsyncClient() as client:
                await client.post(
                    response_url,
                    json={
                        "response_type": "in_channel",
                        "text": message,
                    },
                )
        elif channel_id and should_send_message:
            kwargs: dict[str, Any] = {
                "channel": channel_id,
                "text": message,
                "thread_ts": thread_ts,
            }
            if message_blocks:
                kwargs["blocks"] = message_blocks
            await _run_slack_call(slack_client.chat_postMessage, **kwargs)
    finally:
        if channel_id and message_ts:
            try:
                await _run_slack_call(
                    slack_client.reactions_remove,
                    channel=channel_id,
                    name="loading",
                    timestamp=message_ts,
                )
            except Exception as e:
                logger.warning(f"Failed to remove loading reaction: {e}")

    if not result.get("should_respond") and not response_url:
        return {"status": "ignored", "result": result}

    if result.get("error"):
        return {"status": "error", "message": result.get("error")}
    return {"status": "ok", "result": result}


async def handle_bigquery_approval_bg(payload: dict[str, Any]) -> dict[str, Any]:
    """Background handler for approval/denial callbacks."""
    from data_bolt.tasks.analyst_agent import run_analyst_approval

    channel_id = payload.get("channel_id")
    thread_ts = payload.get("thread_ts")
    response_url = payload.get("response_url")

    result = await to_thread.run_sync(run_analyst_approval, payload)
    should_send_message = bool(result.get("should_respond")) or bool(response_url)
    message = _format_bigquery_response(result) if should_send_message else ""
    message_blocks = (
        result.get("response_blocks") if isinstance(result.get("response_blocks"), list) else None
    )

    if response_url and should_send_message:
        async with httpx.AsyncClient() as client:
            await client.post(
                response_url,
                json={
                    "response_type": "in_channel",
                    "text": message,
                },
            )
    elif channel_id and should_send_message:
        kwargs: dict[str, Any] = {
            "channel": channel_id,
            "text": message,
            "thread_ts": thread_ts,
        }
        if message_blocks:
            kwargs["blocks"] = message_blocks
        await _run_slack_call(slack_client.chat_postMessage, **kwargs)

    if result.get("error"):
        return {"status": "error", "message": result.get("error")}
    return {"status": "ok", "result": result}
