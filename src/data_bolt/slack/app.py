"""Slack Bolt application configuration."""

import logging
import os
from typing import Any

from slack_bolt import App, BoltResponse
from slack_bolt.error import BoltUnhandledRequestError

logger = logging.getLogger(__name__)


def custom_error_handler(
    error: Exception,
    body: dict[str, Any],
    logger: logging.Logger,
) -> BoltResponse:
    """
    Global error handler for Slack Bolt app.

    Catches unhandled exceptions and returns appropriate responses.
    """
    # Handle unhandled requests (no matching handler)
    if isinstance(error, BoltUnhandledRequestError):
        event_type = body.get("event", {}).get("type", "unknown")
        command = body.get("command", "none")
        logger.warning(f"Unhandled Slack request: event_type={event_type}, command={command}")
        return BoltResponse(status=200, body="")

    # Handle other errors
    logger.exception(f"Unhandled Slack error: {error}")
    return BoltResponse(
        status=200,
        body="An error occurred processing your request. Please try again later.",
    )


slack_app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
    process_before_response=True,  # Required for Lambda - must ack before timeout
)

# Register error handler
slack_app.error(custom_error_handler)

# Import handlers AFTER slack_app is defined to avoid circular import
from . import handlers  # noqa: F401, E402 - registers handlers
