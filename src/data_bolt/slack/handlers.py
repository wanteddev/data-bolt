"""Slack event and command handlers."""

import json
import logging
import os
from collections.abc import Mapping
from typing import Any

import boto3
from anyio import to_thread
from botocore.exceptions import ClientError
from slack_bolt.async_app import AsyncAck, AsyncBoltContext, AsyncSay

from data_bolt.tasks.relevance import should_respond_to_message

from .app import slack_app

logger = logging.getLogger(__name__)
type AgentPayload = dict[str, Any]

lambda_client = boto3.client("lambda")
BACKGROUND_FUNCTION_NAME = os.environ.get("SLACK_BG_FUNCTION_NAME", "")


class SlackBackgroundError(Exception):
    """Raised when background Lambda invocation fails."""

    pass


def _should_process_message_event(event: dict[str, Any], text: str) -> bool:
    if event.get("bot_id") or event.get("subtype"):
        return False

    is_thread_followup = bool(event.get("thread_ts")) and event.get("thread_ts") != event.get("ts")
    return should_respond_to_message(
        text=text,
        channel_type=str(event.get("channel_type") or ""),
        is_mention=False,
        is_thread_followup=is_thread_followup,
        channel_id=str(event.get("channel") or ""),
    )


def _build_bigquery_payload(
    event: dict[str, Any], text: str, *, is_mention: bool
) -> "AgentPayload":
    payload: AgentPayload = {
        "user_id": str(event.get("user") or ""),
        "channel_id": str(event.get("channel") or ""),
        "channel_type": str(event.get("channel_type") or ""),
        "thread_ts": str(event.get("thread_ts") or event.get("ts") or ""),
        "message_ts": str(event.get("ts") or ""),
        "is_thread_followup": bool(
            event.get("thread_ts") and event.get("thread_ts") != event.get("ts")
        ),
        "is_mention": is_mention,
        "text": text,
        "team_id": str(event.get("team") or ""),
        "include_thread_history": True,
    }
    return payload


def _extract_action_value(body: dict[str, Any]) -> str:
    actions = body.get("actions")
    if not isinstance(actions, list) or not actions:
        return ""
    first = actions[0]
    if not isinstance(first, dict):
        return ""
    value = first.get("value")
    return str(value) if value is not None else ""


def _build_approval_payload(body: dict[str, Any], *, approved: bool) -> AgentPayload:
    team_raw = body.get("team")
    user_raw = body.get("user")
    channel_raw = body.get("channel")
    container_raw = body.get("container")
    message_raw = body.get("message")

    team = team_raw if isinstance(team_raw, Mapping) else {}
    user = user_raw if isinstance(user_raw, Mapping) else {}
    channel = channel_raw if isinstance(channel_raw, Mapping) else {}
    container = container_raw if isinstance(container_raw, Mapping) else {}
    message = message_raw if isinstance(message_raw, Mapping) else {}

    channel_id = str(channel.get("id") or container.get("channel_id") or "")
    message_ts = str(container.get("message_ts") or message.get("ts") or "")
    thread_ts = str(message.get("thread_ts") or container.get("thread_ts") or message_ts)

    return {
        "approval_request_id": _extract_action_value(body),
        "approved": approved,
        "user_id": str(user.get("id") or ""),
        "team_id": str(team.get("id") or ""),
        "channel_id": channel_id,
        "thread_ts": thread_ts,
        "message_ts": message_ts,
    }


def invoke_background(task_type: str, payload: Mapping[str, Any]) -> None:
    """
    Invoke background Lambda for long-running tasks.

    Args:
        task_type: Type of background task to execute
        payload: Data to pass to background handler

    Raises:
        SlackBackgroundError: If Lambda invocation fails
    """
    if not BACKGROUND_FUNCTION_NAME:
        logger.warning("SLACK_BG_FUNCTION_NAME not set, skipping background invoke")
        return

    try:
        response = lambda_client.invoke(
            FunctionName=BACKGROUND_FUNCTION_NAME,
            InvocationType="Event",  # Async invocation
            Payload=json.dumps({"task_type": task_type, "payload": payload}),
        )

        status_code = response.get("StatusCode", 0)
        if status_code not in (200, 202):
            logger.error(f"Background Lambda returned unexpected status: {status_code}")
            raise SlackBackgroundError(f"Background invocation failed with status {status_code}")

        logger.info(f"Background task '{task_type}' invoked successfully")

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        logger.error(f"Failed to invoke background Lambda: {error_code} - {e}")
        raise SlackBackgroundError(f"Failed to invoke background Lambda: {error_code}") from e


# =============================================================================
# Slash Commands
# =============================================================================


@slack_app.command("/hello")
async def handle_hello_command(ack: AsyncAck, command: dict[str, Any], say: AsyncSay) -> None:
    """
    Example slash command handler.

    For quick responses (< 3 seconds), respond directly.
    """
    await ack()
    user_id = command["user_id"]
    await say(f"Hello <@{user_id}>! :wave:")


# =============================================================================
# Event Handlers
# =============================================================================


@slack_app.event("app_mention")
async def handle_app_mention(
    event: dict[str, Any],
    say: AsyncSay,
    context: AsyncBoltContext,
    ack: AsyncAck,
) -> None:
    """
    Handle when the bot is mentioned in a channel.

    For quick responses, reply directly. For complex tasks, use background processing.
    """
    await ack()
    user_id = event["user"]
    text = event.get("text", "")
    try:
        try:
            await context.client.reactions_add(
                channel=event["channel"], name="loading", timestamp=event["ts"]
            )
        except Exception as e:
            logger.warning(f"Failed to add loading reaction: {e}")
        await to_thread.run_sync(
            invoke_background,
            "bigquery_sql",
            _build_bigquery_payload(event, text, is_mention=True),
        )
    except SlackBackgroundError as e:
        logger.error(f"Background task failed for mention from {user_id}: {e}")
        await say(":x: 요청 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")


@slack_app.event("message")
async def handle_message(
    event: dict[str, Any],
    say: AsyncSay,
    context: AsyncBoltContext,
    ack: AsyncAck,
) -> None:
    """
    Handle direct messages to the bot.

    Only processes DMs (im channel type). Ignores bot messages to prevent loops.
    """
    await ack()
    text = event.get("text", "")
    if not _should_process_message_event(event, text):
        return

    try:
        if event.get("channel_type") != "im":
            try:
                await context.client.reactions_add(
                    channel=event["channel"], name="loading", timestamp=event["ts"]
                )
            except Exception as e:
                logger.warning(f"Failed to add loading reaction: {e}")
        await to_thread.run_sync(
            invoke_background,
            "bigquery_sql",
            _build_bigquery_payload(event, text, is_mention=False),
        )
    except SlackBackgroundError as e:
        logger.error(f"Background task failed for message event: {e}")
        await say(
            ":x: 요청 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            thread_ts=event.get("thread_ts") or event.get("ts"),
        )


@slack_app.action("bq_approve_execute")
async def handle_approve_execute_action(
    ack: AsyncAck,
    body: dict[str, Any],
) -> None:
    """Handle Slack button action for query execution approval."""
    await ack()
    payload = _build_approval_payload(body, approved=True)
    try:
        await to_thread.run_sync(invoke_background, "bigquery_approval", payload)
    except SlackBackgroundError as e:
        logger.error(f"Approval background task failed: {e}")


@slack_app.action("bq_deny_execute")
async def handle_deny_execute_action(
    ack: AsyncAck,
    body: dict[str, Any],
) -> None:
    """Handle Slack button action for query execution denial."""
    await ack()
    payload = _build_approval_payload(body, approved=False)
    try:
        await to_thread.run_sync(invoke_background, "bigquery_approval", payload)
    except SlackBackgroundError as e:
        logger.error(f"Denial background task failed: {e}")
