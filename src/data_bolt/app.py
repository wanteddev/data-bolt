"""Litestar application entrypoint."""

import json
import logging
import time
from typing import Any

from slack_bolt.request import BoltRequest
from data_bolt.slack.app import slack_app

from litestar import Litestar, Request, Response, get, post

logger = logging.getLogger(__name__)

_EVENT_DEDUPE_TTL_SECONDS = 300
_event_dedupe_cache: dict[str, float] = {}


def _is_duplicate_event(event_id: str) -> bool:
    now = time.time()
    expires_at = _event_dedupe_cache.get(event_id)
    if expires_at and expires_at > now:
        return True
    _event_dedupe_cache[event_id] = now + _EVENT_DEDUPE_TTL_SECONDS
    if len(_event_dedupe_cache) > 1000:
        expired = [k for k, v in _event_dedupe_cache.items() if v <= now]
        for key in expired:
            _event_dedupe_cache.pop(key, None)
    return False


@get("/")
async def health() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"ok": "web"}


@get("/healthz")
async def healthz() -> dict[str, str]:
    """Lightweight health endpoint for load balancers."""
    return {"status": "healthy"}



@post("/slack/events")
async def handle_slack_events(request: Request) -> Response:
    """
    Handle Slack events (slash commands, events API, interactions).

    This endpoint receives all Slack webhook requests and routes them
    to the appropriate handlers via slack_bolt.
    """

    try:
        body = await request.body()
        headers = {k: v for k, v in request.headers.items()}
        retry_num = next(
            (v for k, v in headers.items() if k.lower() == "x-slack-retry-num"),
            None,
        )
        # Skip retries to avoid duplicate processing
        if retry_num:
            return Response(
                content="ok",
                status_code=200,
                headers={"Content-Type": "text/plain"},
            )
        try:
            payload = json.loads(body.decode("utf-8"))
            event_id = payload.get("event_id")
            if isinstance(event_id, str) and event_id and _is_duplicate_event(event_id):
                return Response(
                    content="ok",
                    status_code=200,
                    headers={"Content-Type": "text/plain"},
                )
        except Exception:
            pass

        bolt_request = BoltRequest(body=body.decode(), headers=headers)
        bolt_response = slack_app.dispatch(bolt_request)

        # Convert bolt headers to simple dict (bolt uses dict[str, list[str]])
        response_headers: dict[str, str] = {}
        if bolt_response.headers:
            for key, values in bolt_response.headers.items():
                if isinstance(values, list):
                    response_headers[key] = values[0] if values else ""
                else:
                    response_headers[key] = str(values)

        return Response(
            content=bolt_response.body or "",
            status_code=bolt_response.status,
            headers=response_headers,
        )

    except Exception as e:
        logger.exception(f"Critical error in Slack events handler: {e}")
        # Return 200 to prevent Slack from retrying
        # Slack will retry 3xx/4xx/5xx responses, causing duplicate processing
        return Response(
            content="Internal error occurred",
            status_code=200,
            headers={"Content-Type": "text/plain"},
        )


@post("/slack/background")
async def handle_slack_background(data: dict[str, Any]) -> dict[str, Any]:
    """
    Handle background task processing for Slack.

    This endpoint is invoked asynchronously by the main handler
    for long-running tasks that exceed Slack's 3-second timeout.
    """
    from data_bolt.slack.background import handle_bigquery_sql_bg, process_background_task

    if data.get("task_type") in {"bigquery_sql", "build_bigquery"}:
        payload = data.get("payload", {})
        return await handle_bigquery_sql_bg(payload)

    return await process_background_task(data)


app = Litestar(
    route_handlers=[
        health,
        healthz,

        handle_slack_events,
        handle_slack_background,
    ],
    debug=False,
)
