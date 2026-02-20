"""DynamoDB-backed thread memory store for analyst agent history."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _coerce_int(value: Any, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _is_conditional_check_failed(exc: Exception) -> bool:
    if not isinstance(exc, ClientError):
        return False
    return exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException"


@dataclass(frozen=True)
class ThreadKey:
    """Primary key components for thread-scoped memory."""

    team_id: str
    channel_id: str
    thread_ts: str

    @property
    def pk(self) -> str:
        return f"THREAD#{self.team_id}#{self.channel_id}#{self.thread_ts}"

    @property
    def is_valid(self) -> bool:
        return bool(self.team_id and self.channel_id and self.thread_ts)


@dataclass
class ThreadTurn:
    """Persisted per-turn delta messages."""

    turn: int
    new_messages_json: str
    usage_prompt_tokens: int | None
    created_at: int
    expires_at: int


@dataclass
class ThreadSummary:
    """Persisted semantic summary memory."""

    version: int
    summary_text: str
    summary_struct_json: str
    source_turn_start: int
    source_turn_end: int
    created_at: int
    expires_at: int


@dataclass
class ThreadMemoryState:
    """Thread memory state loaded from DynamoDB."""

    key: ThreadKey
    version: int
    latest_turn: int
    summary_version: int
    summary_last_turn: int
    prompt_tokens_ema: float | None
    turns: list[ThreadTurn]
    summary: ThreadSummary | None

    @property
    def exists(self) -> bool:
        return bool(self.latest_turn > 0 or self.summary is not None)


@dataclass
class AppendTurnResult:
    """Result after appending one new turn."""

    turn: int
    version: int
    prompt_tokens_ema: float | None


class DynamoThreadMemoryStore:
    """DynamoDB store for thread memory snapshots/deltas."""

    def __init__(self, table_name: str) -> None:
        if not table_name.strip():
            raise ValueError("BIGQUERY_THREAD_MEMORY_DDB_TABLE must be configured")
        self._table = boto3.resource("dynamodb").Table(table_name)
        self._turn_ttl_seconds = _env_int("BIGQUERY_THREAD_MEMORY_TURN_TTL_SECONDS", 7 * 24 * 3600)
        self._meta_ttl_seconds = _env_int("BIGQUERY_THREAD_MEMORY_META_TTL_SECONDS", 30 * 24 * 3600)

    @staticmethod
    def from_env() -> DynamoThreadMemoryStore:
        table = os.getenv("BIGQUERY_THREAD_MEMORY_DDB_TABLE", "").strip()
        return DynamoThreadMemoryStore(table)

    def has_state(self, key: ThreadKey) -> bool:
        if not key.is_valid:
            return False
        response = self._table.query(
            KeyConditionExpression=Key("PK").eq(key.pk),
            Limit=1,
        )
        items = response.get("Items")
        return bool(items)

    def load_state(self, key: ThreadKey) -> ThreadMemoryState | None:
        if not key.is_valid:
            return None

        now = int(time.time())
        items: list[dict[str, Any]] = []
        exclusive_start_key: dict[str, Any] | None = None
        while True:
            params: dict[str, Any] = {
                "KeyConditionExpression": Key("PK").eq(key.pk),
            }
            if exclusive_start_key is not None:
                params["ExclusiveStartKey"] = exclusive_start_key
            response = self._table.query(**params)
            chunk = response.get("Items")
            items.extend(chunk or [])
            raw_next = response.get("LastEvaluatedKey")
            if not raw_next:
                break
            exclusive_start_key = raw_next

        if not items:
            return None

        meta_item: dict[str, Any] | None = None
        turns: list[ThreadTurn] = []
        summaries: list[ThreadSummary] = []

        for item in items:
            expires_at = _coerce_int(item.get("expires_at"))
            if expires_at and expires_at <= now:
                continue
            sk = str(item.get("SK") or "")
            if sk == "META":
                meta_item = item
                continue
            if sk.startswith("TURN#"):
                turn = _coerce_int(sk.removeprefix("TURN#"))
                if turn <= 0:
                    continue
                turns.append(
                    ThreadTurn(
                        turn=turn,
                        new_messages_json=str(item.get("new_messages_json") or "[]"),
                        usage_prompt_tokens=(
                            _coerce_int(item.get("usage_prompt_tokens"))
                            if item.get("usage_prompt_tokens") is not None
                            else None
                        ),
                        created_at=_coerce_int(item.get("created_at")),
                        expires_at=expires_at,
                    )
                )
                continue
            if sk.startswith("SUMMARY#"):
                version = _coerce_int(sk.removeprefix("SUMMARY#"))
                if version <= 0:
                    continue
                summaries.append(
                    ThreadSummary(
                        version=version,
                        summary_text=str(item.get("summary_text") or ""),
                        summary_struct_json=str(item.get("summary_struct_json") or "{}"),
                        source_turn_start=_coerce_int(item.get("source_turn_start")),
                        source_turn_end=_coerce_int(item.get("source_turn_end")),
                        created_at=_coerce_int(item.get("created_at")),
                        expires_at=expires_at,
                    )
                )

        turns.sort(key=lambda t: t.turn)
        summaries.sort(key=lambda s: s.version)

        latest_turn = turns[-1].turn if turns else 0
        version = 0
        summary_version = summaries[-1].version if summaries else 0
        summary_last_turn = summaries[-1].source_turn_end if summaries else 0
        prompt_tokens_ema: float | None = None
        if meta_item is not None:
            version = _coerce_int(meta_item.get("version"))
            latest_turn = _coerce_int(meta_item.get("latest_turn"), latest_turn)
            summary_version = _coerce_int(meta_item.get("summary_version"), summary_version)
            summary_last_turn = _coerce_int(meta_item.get("summary_last_turn"), summary_last_turn)
            prompt_tokens_ema = _coerce_float(meta_item.get("prompt_tokens_ema"))

        summary = next((s for s in reversed(summaries) if s.version == summary_version), None)
        if summary is None and summaries:
            summary = summaries[-1]
            summary_version = summary.version
            summary_last_turn = summary.source_turn_end

        state = ThreadMemoryState(
            key=key,
            version=version,
            latest_turn=latest_turn,
            summary_version=summary_version,
            summary_last_turn=summary_last_turn,
            prompt_tokens_ema=prompt_tokens_ema,
            turns=turns,
            summary=summary,
        )
        return state if state.exists else None

    def append_turn(
        self,
        *,
        key: ThreadKey,
        new_messages_json: str,
        prompt_tokens_ema: float | None,
        usage_prompt_tokens: int | None,
    ) -> AppendTurnResult:
        if not key.is_valid:
            raise ValueError("Invalid thread key")

        now = int(time.time())
        meta_exp = now + self._meta_ttl_seconds
        turn_exp = now + self._turn_ttl_seconds
        prompt_tokens_ema_value = Decimal(str(prompt_tokens_ema or 0.0))

        for _ in range(2):
            update_response = self._table.update_item(
                Key={"PK": key.pk, "SK": "META"},
                UpdateExpression=(
                    "SET #version = if_not_exists(#version, :zero) + :one, "
                    "#latest_turn = if_not_exists(#latest_turn, :zero) + :one, "
                    "#summary_version = if_not_exists(#summary_version, :zero), "
                    "#summary_last_turn = if_not_exists(#summary_last_turn, :zero), "
                    "#prompt_tokens_ema = :prompt_tokens_ema, "
                    "#updated_at = :now, "
                    "#expires_at = :meta_exp"
                ),
                ExpressionAttributeNames={
                    "#version": "version",
                    "#latest_turn": "latest_turn",
                    "#summary_version": "summary_version",
                    "#summary_last_turn": "summary_last_turn",
                    "#prompt_tokens_ema": "prompt_tokens_ema",
                    "#updated_at": "updated_at",
                    "#expires_at": "expires_at",
                },
                ExpressionAttributeValues={
                    ":zero": 0,
                    ":one": 1,
                    ":prompt_tokens_ema": prompt_tokens_ema_value,
                    ":now": now,
                    ":meta_exp": meta_exp,
                },
                ReturnValues="ALL_NEW",
            )

            attrs = update_response.get("Attributes") or {}
            next_turn = _coerce_int(attrs.get("latest_turn"))
            version = _coerce_int(attrs.get("version"))
            if next_turn <= 0:
                raise RuntimeError("Invalid latest_turn in thread memory meta")

            try:
                self._table.put_item(
                    Item={
                        "PK": key.pk,
                        "SK": f"TURN#{next_turn:08d}",
                        "turn": next_turn,
                        "new_messages_json": new_messages_json,
                        "usage_prompt_tokens": usage_prompt_tokens,
                        "created_at": now,
                        "expires_at": turn_exp,
                    },
                    ConditionExpression="attribute_not_exists(PK) AND attribute_not_exists(SK)",
                )
                return AppendTurnResult(
                    turn=next_turn,
                    version=version,
                    prompt_tokens_ema=_coerce_float(attrs.get("prompt_tokens_ema")),
                )
            except Exception as exc:
                if _is_conditional_check_failed(exc):
                    continue
                raise
        raise RuntimeError("Failed to append thread turn after retry")

    def save_summary_with_cas(
        self,
        *,
        key: ThreadKey,
        expected_version: int,
        expected_summary_version: int,
        summary_text: str,
        summary_struct_json: str,
        source_turn_start: int,
        source_turn_end: int,
        prompt_tokens_ema: float | None,
    ) -> bool:
        if not key.is_valid:
            return False

        now = int(time.time())
        meta_exp = now + self._meta_ttl_seconds
        summary_version = expected_summary_version + 1
        summary_sk = f"SUMMARY#{summary_version:08d}"

        try:
            self._table.put_item(
                Item={
                    "PK": key.pk,
                    "SK": summary_sk,
                    "summary_text": summary_text,
                    "summary_struct_json": summary_struct_json,
                    "source_turn_start": source_turn_start,
                    "source_turn_end": source_turn_end,
                    "created_at": now,
                    "expires_at": meta_exp,
                },
                ConditionExpression="attribute_not_exists(PK) AND attribute_not_exists(SK)",
            )
        except Exception:
            return False

        expression_names = {
            "#version": "version",
            "#summary_version": "summary_version",
            "#summary_last_turn": "summary_last_turn",
            "#updated_at": "updated_at",
            "#expires_at": "expires_at",
        }
        expression_values: dict[str, Any] = {
            ":expected_version": expected_version,
            ":expected_summary_version": expected_summary_version,
            ":one": 1,
            ":new_summary_version": summary_version,
            ":source_turn_end": source_turn_end,
            ":now": now,
            ":meta_exp": meta_exp,
        }
        update_expression = (
            "SET #version = #version + :one, "
            "#summary_version = :new_summary_version, "
            "#summary_last_turn = :source_turn_end, "
            "#updated_at = :now, "
            "#expires_at = :meta_exp"
        )

        if prompt_tokens_ema is not None:
            expression_names["#prompt_tokens_ema"] = "prompt_tokens_ema"
            expression_values[":prompt_tokens_ema"] = Decimal(str(prompt_tokens_ema))
            update_expression += ", #prompt_tokens_ema = :prompt_tokens_ema"

        try:
            self._table.update_item(
                Key={"PK": key.pk, "SK": "META"},
                ConditionExpression=(
                    "#version = :expected_version AND #summary_version = :expected_summary_version"
                ),
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expression_names,
                ExpressionAttributeValues=expression_values,
            )
            return True
        except Exception:
            self._table.delete_item(Key={"PK": key.pk, "SK": summary_sk})
            return False
