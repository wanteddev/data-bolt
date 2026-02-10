"""Runtime helpers for botctl backend and policy decisions."""

from __future__ import annotations

import os
from typing import Literal, cast

import typer

CheckpointBackend = Literal["memory", "postgres", "dynamodb"]


def resolve_checkpoint_backend() -> CheckpointBackend:
    """Resolve checkpoint backend from env with safe fallback."""
    raw = (os.getenv("LANGGRAPH_CHECKPOINT_BACKEND") or "").strip().lower()
    if raw in {"memory", "postgres", "dynamodb"}:
        return cast(CheckpointBackend, raw)
    return "memory"


def guard_thread_ts_with_backend(
    *,
    thread_ts: str | None,
    backend: CheckpointBackend,
    command: str,
) -> None:
    """Apply command-specific thread persistence policy."""
    if command == "simulate" and thread_ts and backend == "memory":
        raise typer.BadParameter(
            (
                "`--thread-ts` 지속 대화는 memory backend에서 지원되지 않습니다. "
                "`LANGGRAPH_CHECKPOINT_BACKEND=postgres` 또는 "
                "`LANGGRAPH_CHECKPOINT_BACKEND=dynamodb`를 설정하거나 `botctl chat`을 사용하세요."
            ),
            param_hint="--thread-ts",
        )
