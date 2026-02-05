"""Slack event and command handlers."""

import json
import logging
import os
from typing import Any

import boto3
from botocore.exceptions import ClientError
from anyio import to_thread
from slack_bolt.async_app import AsyncAck, AsyncBoltContext, AsyncSay

from .app import slack_app

logger = logging.getLogger(__name__)

lambda_client = boto3.client("lambda")
BACKGROUND_FUNCTION_NAME = os.environ.get("SLACK_BG_FUNCTION_NAME", "")


class SlackBackgroundError(Exception):
    """Raised when background Lambda invocation fails."""

    pass


def invoke_background(task_type: str, payload: dict[str, Any]) -> None:
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
async def handle_hello_command(
    ack: AsyncAck, command: dict[str, Any], say: AsyncSay
) -> None:
    """
    Example slash command handler.

    For quick responses (< 3 seconds), respond directly.
    """
    await ack()
    user_id = command["user_id"]
    await say(f"Hello <@{user_id}>! :wave:")


@slack_app.command("/longtask")
async def handle_long_task_command(
    ack: AsyncAck,
    command: dict[str, Any],
    say: AsyncSay,
) -> None:
    """
    Example slash command handler.

    For quick processing, handle directly after ack.
    """
    # Immediately acknowledge to avoid 3-second timeout
    await ack("Processing your request... :hourglass_flowing_sand:")

    # ==========================================================================
    # YOUR LONG-RUNNING LOGIC HERE
    # Keep it fast; move to SlackBgFunction if it can exceed ~3 seconds.
    # ==========================================================================
    text = command.get("text", "")
    result = f"Processed '{text}' successfully!"
    await say(f":white_check_mark: {result}")


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
            {
                "user_id": user_id,
                "channel_id": event["channel"],
                "thread_ts": event.get("thread_ts") or event.get("ts"),
                "message_ts": event.get("ts"),
                "text": text,
            },
        )
    except SlackBackgroundError as e:
        logger.error(f"Background task failed for mention from {user_id}: {e}")
        await say(":x: 요청 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")


@slack_app.event("message")
async def handle_message(
    event: dict[str, Any],
    say: AsyncSay,
    ack: AsyncAck,
) -> None:
    """
    Handle direct messages to the bot.

    Only processes DMs (im channel type). Ignores bot messages to prevent loops.
    """
    await ack()
    # Ignore bot messages and message subtypes (edits, deletes, etc.)
    if event.get("bot_id") or event.get("subtype"):
        return

    # Only respond to direct messages
    channel_type = event.get("channel_type", "")
    if channel_type != "im":
        return

    text = event.get("text", "")

    # ==========================================================================
    # YOUR DM LOGIC HERE (keep it fast; offload if it grows)
    # ==========================================================================
    result = f"I processed your message: '{text}'"
    await say(result)
