"""Simulation command for botctl."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from dotenv import load_dotenv

from data_bolt.botctl.runtime import guard_thread_ts_with_backend, resolve_checkpoint_backend
from data_bolt.botctl.types import SimulationCase, SimulationRun

type TraceCallback = Callable[[str, str], None]

DEFAULT_CASES: dict[str, SimulationCase] = {
    "greeting": {"name": "greeting", "text": "안녕하세요"},
    "smalltalk": {"name": "smalltalk", "text": "오늘 날씨 어때?"},
    "sql_gen": {"name": "sql_gen", "text": "지난주 가입자 수 쿼리 만들어줘"},
    "sql_exec": {
        "name": "sql_exec",
        "text": "이 쿼리 실행해줘\n```sql\nSELECT 1 AS value;\n```",
    },
}


def _build_payload(
    *,
    text: str,
    channel_type: str,
    mention: bool,
    thread_followup: bool,
    thread_ts: str | None = None,
) -> dict[str, Any]:
    message_ts = str(time.time())
    return {
        "user_id": "U_LOCAL",
        "team_id": "local",
        "channel_id": "DLOCAL",
        "channel_type": channel_type,
        "thread_ts": thread_ts or message_ts,
        "message_ts": message_ts,
        "is_mention": mention,
        "is_thread_followup": thread_followup,
        "include_thread_history": False,
        "text": text,
    }


def _resolve_from_file(path: Path) -> dict[str, SimulationCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        resolved: dict[str, SimulationCase] = {}
        for case_name, value in raw.items():
            if isinstance(value, dict):
                case = cast(SimulationCase, dict(value))
                case["name"] = str(case.get("name") or case_name)
                resolved[str(case_name)] = case
        return resolved

    if isinstance(raw, list):
        resolved = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if isinstance(name, str) and name:
                resolved[name] = cast(SimulationCase, item)
        return resolved

    raise typer.BadParameter("Case file must be a JSON object or array.")


def _validate_single_input(text: str | None, case: str | None, file: Path | None) -> None:
    provided_count = sum(1 for v in (text, case, file) if v)
    if provided_count != 1:
        raise typer.BadParameter("Provide exactly one of --text, --case, or --file.")


def _render_summary(run: SimulationRun) -> str:
    result = run["result"]
    lines = [
        f"mode: {run['mode']}",
        f"action: {result.get('action')}",
        f"should_respond: {result.get('should_respond')}",
        f"candidate_sql: {result.get('candidate_sql')}",
        "response_text:",
        str(result.get("response_text") or ""),
    ]
    return "\n".join(lines)


async def _run_via_background(payload: dict[str, Any]) -> dict[str, Any]:
    from data_bolt.slack import background

    async def _noop_run_slack_call(_func: Any, /, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    original_run_slack_call = background._run_slack_call
    background._run_slack_call = cast(Any, _noop_run_slack_call)
    try:
        outcome = await background.handle_bigquery_sql_bg(payload)
    finally:
        background._run_slack_call = original_run_slack_call

    nested = outcome.get("result")
    if isinstance(nested, dict):
        return nested
    return {
        "action": None,
        "should_respond": False,
        "candidate_sql": None,
        "response_text": json.dumps(outcome, ensure_ascii=False),
    }


def _run_single(payload: dict[str, Any], via_background: bool) -> SimulationRun:
    if via_background:
        background_result = asyncio.run(_run_via_background(payload))
        return {"mode": "background", "payload": payload, "result": background_result}

    from data_bolt.tasks.analyst_agent import run_analyst_turn

    direct_result = run_analyst_turn(payload)
    return {"mode": "direct", "payload": payload, "result": direct_result}


def _trace_stdout_callback() -> TraceCallback:
    def _emit(node: str, reason: str) -> None:
        typer.echo(f"[trace] {node}: {reason}")

    return _emit


def _trace_reason_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    raw_trace = result.get("trace")
    if not isinstance(raw_trace, list):
        return []

    resolved: list[dict[str, Any]] = []
    for entry in raw_trace:
        if not isinstance(entry, dict):
            continue
        node = str(entry.get("node") or "").strip()
        reason = str(entry.get("reason") or "").strip()
        if not node:
            continue
        resolved.append({"node": node, "reason": reason or "실행 단계 완료"})
    return resolved


def _run_direct_persistent(payload: dict[str, Any], trace_enabled: bool) -> SimulationRun:
    printed_live_trace = False
    run: SimulationRun
    if trace_enabled:
        from data_bolt.tasks.analyst_agent import run_analyst_turn

        run = {
            "mode": "direct",
            "payload": payload,
            "result": run_analyst_turn(payload, trace_callback=_trace_stdout_callback()),
        }
        printed_live_trace = True
    else:
        run = _run_single(payload, False)
    trace = _trace_reason_from_result(run["result"])
    run["trace"] = trace
    if trace_enabled and not printed_live_trace:
        for entry in trace:
            typer.echo(f"[trace] {entry['node']}: {entry['reason']}")
    return run


def _run_direct_with_trace(payload: dict[str, Any], trace_enabled: bool) -> SimulationRun:
    printed_live_trace = False
    run: SimulationRun
    if trace_enabled:
        from data_bolt.tasks.analyst_agent import run_analyst_turn

        run = {
            "mode": "direct",
            "payload": payload,
            "result": run_analyst_turn(payload, trace_callback=_trace_stdout_callback()),
        }
        printed_live_trace = True
    else:
        run = _run_single(payload, False)
    trace = _trace_reason_from_result(run["result"])
    run["trace"] = trace
    if trace_enabled and not printed_live_trace:
        for entry in trace:
            typer.echo(f"[trace] {entry['node']}: {entry['reason']}")
    return run


def simulate_command(
    text: Annotated[str | None, typer.Option("--text", help="Message text to simulate.")] = None,
    case: Annotated[str | None, typer.Option("--case", help="Predefined case name to run.")] = None,
    file: Annotated[Path | None, typer.Option("--file", help="JSON case file path.")] = None,
    channel_type: Annotated[str, typer.Option("--channel-type", help="Slack channel type.")] = "im",
    mention: Annotated[
        bool, typer.Option("--mention/--no-mention", help="Set mention flag.")
    ] = False,
    thread_followup: Annotated[
        bool,
        typer.Option(
            "--thread-followup/--no-thread-followup",
            help="Set thread followup flag.",
        ),
    ] = False,
    via_background: Annotated[
        bool,
        typer.Option("--via-background/--direct", help="Run through background handler path."),
    ] = False,
    thread_ts: Annotated[
        str | None,
        typer.Option("--thread-ts", help="Reuse a fixed thread id for multi-turn continuity."),
    ] = None,
    trace: Annotated[
        bool,
        typer.Option("--trace/--no-trace", help="Show live runtime trace while running."),
    ] = True,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON output.")] = False,
    env_file: Annotated[
        Path, typer.Option("--env-file", help="Load environment variables from this .env file.")
    ] = Path(".env"),
) -> None:
    """Run local simulation without deploying or sending Slack events."""
    load_dotenv(dotenv_path=env_file, override=False, encoding="utf-8")
    backend = resolve_checkpoint_backend()
    guard_thread_ts_with_backend(thread_ts=thread_ts, backend=backend, command="simulate")
    _validate_single_input(text, case, file)
    runs: list[SimulationRun] = []

    if text:
        payload = _build_payload(
            text=text,
            channel_type=channel_type,
            mention=mention,
            thread_followup=thread_followup,
            thread_ts=thread_ts,
        )
        if via_background:
            if trace and not as_json:
                typer.echo("[trace] background: /slack/background 경로로 실행합니다.")
            runs.append(_run_single(payload, True))
        else:
            if thread_ts:
                runs.append(_run_direct_persistent(payload, trace and not as_json))
            else:
                runs.append(_run_direct_with_trace(payload, trace and not as_json))

    elif case:
        source_cases = DEFAULT_CASES
        selected = source_cases.get(case)
        if not selected:
            available = ", ".join(sorted(source_cases))
            raise typer.BadParameter(f"Unknown case '{case}'. Available: {available}")
        payload = _build_payload(
            text=str(selected.get("text") or ""),
            channel_type=str(selected.get("channel_type") or channel_type),
            mention=bool(selected.get("is_mention", mention)),
            thread_followup=bool(selected.get("is_thread_followup", thread_followup)),
            thread_ts=thread_ts,
        )
        resolved_mode = bool(selected.get("via_background", via_background))
        if resolved_mode:
            if trace and not as_json:
                typer.echo("[trace] background: /slack/background 경로로 실행합니다.")
            runs.append(_run_single(payload, True))
        else:
            if thread_ts:
                runs.append(_run_direct_persistent(payload, trace and not as_json))
            else:
                runs.append(_run_direct_with_trace(payload, trace and not as_json))

    elif file:
        file_cases = _resolve_from_file(file)
        if not file_cases:
            raise typer.BadParameter("No valid cases found in file.")
        for _, selected in sorted(file_cases.items()):
            payload = _build_payload(
                text=str(selected.get("text") or ""),
                channel_type=str(selected.get("channel_type") or channel_type),
                mention=bool(selected.get("is_mention", mention)),
                thread_followup=bool(selected.get("is_thread_followup", thread_followup)),
                thread_ts=thread_ts,
            )
            resolved_mode = bool(selected.get("via_background", via_background))
            if resolved_mode:
                if trace and not as_json:
                    typer.echo("[trace] background: /slack/background 경로로 실행합니다.")
                runs.append(_run_single(payload, True))
            else:
                if thread_ts:
                    runs.append(_run_direct_persistent(payload, trace and not as_json))
                else:
                    runs.append(_run_direct_with_trace(payload, trace and not as_json))

    if as_json:
        typer.echo(
            json.dumps(
                runs if len(runs) > 1 else runs[0], ensure_ascii=False, indent=2, default=str
            )
        )
        return

    for index, run in enumerate(runs):
        if len(runs) > 1:
            typer.echo(f"[case {index + 1}]")
        typer.echo(_render_summary(run))
        if index < len(runs) - 1:
            typer.echo("")
