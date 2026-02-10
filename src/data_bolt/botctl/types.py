"""Shared types for botctl CLI."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class SimulationCase(TypedDict, total=False):
    """A reusable simulation case for botctl."""

    name: str
    text: str
    channel_type: str
    is_mention: bool
    is_thread_followup: bool
    via_background: bool


class SimulationRun(TypedDict):
    """Execution metadata and result for one simulation."""

    mode: str
    payload: dict[str, Any]
    result: dict[str, Any]
    trace: NotRequired[list[dict[str, Any]]]
