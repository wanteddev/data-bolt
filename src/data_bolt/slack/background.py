"""Background task processor for Slack bot."""

import json
import logging
import os
import traceback
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, ParamSpec, TypeVar, cast

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
    if validation.get("success"):
        bytes_processed = validation.get("total_bytes_processed")
        estimated_cost = validation.get("estimated_cost_usd")
        cost_summary = (
            f" (bytes={bytes_processed}, est_cost_usd={estimated_cost})"
            if bytes_processed is not None or estimated_cost is not None
            else ""
        )
        if validation.get("refined"):
            return f":white_check_mark: Dry-run passed after refine.{cost_summary}"
        return f":white_check_mark: Dry-run passed.{cost_summary}"
    if validation.get("error"):
        return f":warning: Dry-run failed: {validation.get('error')}"
    return None


def _format_bigquery_response(result: Mapping[str, Any]) -> str:
    if result.get("response_text"):
        return str(result["response_text"])

    if result.get("error"):
        return f":x: {result['error']}"

    generation_result = result.get("generation_result")
    if not isinstance(generation_result, dict):
        generation_result = result

    sql = _extract_sql_from_result(result)
    explanation = (generation_result.get("answer_structured") or {}).get("explanation")
    parts: list[str] = []
    if sql:
        parts.append(f"```sql\n{sql}\n```")
    if isinstance(explanation, str) and explanation.strip():
        parts.append(explanation.strip())
    validation_raw = result.get("validation")
    validation_data = validation_raw if isinstance(validation_raw, dict) else {}
    validation = _format_validation_summary(validation_data)
    if validation:
        parts.append(validation)
    execution = result.get("execution") if isinstance(result.get("execution"), dict) else {}
    if execution:
        if execution.get("success"):
            preview = execution.get("preview_rows") or []
            preview_text = json.dumps(preview, ensure_ascii=False, default=str)[:1200]
            parts.append(
                ":rocket: Query executed."
                f"\njob_id={execution.get('job_id')}, row_count={execution.get('row_count')}"
                f"\npreview={preview_text}"
            )
        elif execution.get("error"):
            parts.append(f":warning: Query execution skipped: {execution.get('error')}")
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
    from data_bolt.tasks.bigquery_agent import AgentPayload, run_bigquery_agent

    response_url = payload.get("response_url")
    channel_id = payload.get("channel_id")
    thread_ts = payload.get("thread_ts")
    message_ts = payload.get("message_ts")

    logger.info("Background processing BigQuery SQL request")

    include_history = payload.get("include_thread_history", True)
    if include_history and not payload.get("history"):
        history = await _fetch_thread_history(channel_id, thread_ts, message_ts)
        if history:
            payload = {**payload, "history": history}

    typed_payload = cast(AgentPayload, payload)
    result = await to_thread.run_sync(run_bigquery_agent, typed_payload)
    should_send_message = bool(result.get("should_respond")) or bool(response_url)
    message = _format_bigquery_response(result) if should_send_message else ""

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
            await _run_slack_call(
                slack_client.chat_postMessage,
                channel=channel_id,
                text=message,
                thread_ts=thread_ts,
            )
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
