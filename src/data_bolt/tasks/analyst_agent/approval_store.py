"""DynamoDB-backed storage for deferred approval contexts."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import boto3


@dataclass
class ApprovalContext:
    """Serialized context required to resume deferred tool execution."""

    approval_request_id: str
    requester_user_id: str
    channel_id: str
    thread_ts: str
    team_id: str
    tool_call_ids: list[str]
    deferred_metadata: dict[str, dict[str, Any]]
    session_messages_json: str
    created_at: int
    expires_at: int


class DynamoApprovalStore:
    """Minimal DynamoDB table wrapper for approval contexts."""

    def __init__(self, table_name: str) -> None:
        if not table_name.strip():
            raise ValueError("BIGQUERY_APPROVAL_DDB_TABLE must be configured")
        dynamodb = boto3.resource("dynamodb")
        self._table = dynamodb.Table(table_name)

    def save(self, context: ApprovalContext) -> None:
        self._table.put_item(
            Item={
                "approval_request_id": context.approval_request_id,
                "requester_user_id": context.requester_user_id,
                "channel_id": context.channel_id,
                "thread_ts": context.thread_ts,
                "team_id": context.team_id,
                "tool_call_ids": context.tool_call_ids,
                "deferred_metadata": context.deferred_metadata,
                "session_messages_json": context.session_messages_json,
                "created_at": context.created_at,
                "expires_at": context.expires_at,
            }
        )

    def load(self, approval_request_id: str) -> ApprovalContext | None:
        response = self._table.get_item(Key={"approval_request_id": approval_request_id})
        item = response.get("Item")
        if not isinstance(item, dict):
            return None

        raw_tool_call_ids = item.get("tool_call_ids")
        tool_call_ids = (
            [str(v) for v in raw_tool_call_ids if isinstance(v, str)]
            if isinstance(raw_tool_call_ids, list)
            else []
        )
        raw_meta = item.get("deferred_metadata")
        deferred_metadata = (
            raw_meta
            if isinstance(raw_meta, dict)
            and all(isinstance(value, dict) for value in raw_meta.values())
            else {}
        )

        return ApprovalContext(
            approval_request_id=str(item.get("approval_request_id") or ""),
            requester_user_id=str(item.get("requester_user_id") or ""),
            channel_id=str(item.get("channel_id") or ""),
            thread_ts=str(item.get("thread_ts") or ""),
            team_id=str(item.get("team_id") or ""),
            tool_call_ids=tool_call_ids,
            deferred_metadata=deferred_metadata,
            session_messages_json=str(item.get("session_messages_json") or "[]"),
            created_at=_coerce_int(item.get("created_at")),
            expires_at=_coerce_int(item.get("expires_at")),
        )

    def delete(self, approval_request_id: str) -> None:
        self._table.delete_item(Key={"approval_request_id": approval_request_id})

    @staticmethod
    def from_env() -> DynamoApprovalStore:
        table = os.getenv("BIGQUERY_APPROVAL_DDB_TABLE", "").strip()
        return DynamoApprovalStore(table)


def build_approval_context_timestamps() -> tuple[int, int]:
    """Build `(created_at, expires_at)` timestamps with configured TTL."""
    created_at = int(time.time())
    ttl = os.getenv("BIGQUERY_APPROVAL_TTL_SECONDS", "3600").strip()
    try:
        ttl_seconds = int(ttl)
    except ValueError:
        ttl_seconds = 3600
    if ttl_seconds <= 0:
        ttl_seconds = 3600
    return created_at, created_at + ttl_seconds


def _coerce_int(value: Any) -> int:
    if isinstance(value, (int, Decimal)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0
