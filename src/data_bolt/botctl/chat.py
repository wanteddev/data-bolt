"""Interactive multi-turn chat command for botctl."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from data_bolt.botctl.runtime import resolve_checkpoint_backend
from data_bolt.botctl.simulate import _build_payload, _run_direct_persistent

_EXIT_WORDS = {"exit", "quit", "/exit"}


def chat_command(
    thread_ts: Annotated[
        str | None,
        typer.Option("--thread-ts", help="Use fixed thread id across turns."),
    ] = None,
    channel_type: Annotated[str, typer.Option("--channel-type", help="Slack channel type.")] = "im",
    mention: Annotated[
        bool, typer.Option("--mention/--no-mention", help="Set mention flag.")
    ] = False,
    thread_followup: Annotated[
        bool,
        typer.Option("--thread-followup/--no-thread-followup", help="Set thread followup flag."),
    ] = False,
    trace: Annotated[
        bool,
        typer.Option("--trace/--no-trace", help="Show live graph node trace while running."),
    ] = True,
    as_json: Annotated[bool, typer.Option("--json", help="Print turn results as JSON.")] = False,
    env_file: Annotated[
        Path, typer.Option("--env-file", help="Load environment variables from this .env file.")
    ] = Path(".env"),
) -> None:
    """Run interactive multi-turn chat in a single process."""
    load_dotenv(dotenv_path=env_file, override=False, encoding="utf-8")
    selected_backend = resolve_checkpoint_backend()
    session_thread_ts = thread_ts or str(time.time())

    if not as_json:
        runtime_mode = os.getenv("BIGQUERY_AGENT_RUNTIME_MODE", "loop")
        typer.echo(
            f"[chat] backend={selected_backend}, runtime_mode={runtime_mode}, "
            f"thread_ts={session_thread_ts}. "
            "종료하려면 exit, quit, /exit 또는 Ctrl-D"
        )

    turn = 1
    while True:
        try:
            user_text = typer.prompt("you").strip()
        except (EOFError, KeyboardInterrupt):
            if not as_json:
                typer.echo("")
                typer.echo("[chat] session ended.")
            break

        if not user_text:
            continue
        if user_text.lower() in _EXIT_WORDS:
            if not as_json:
                typer.echo("[chat] bye.")
            break

        payload = _build_payload(
            text=user_text,
            channel_type=channel_type,
            mention=mention,
            thread_followup=thread_followup,
            thread_ts=session_thread_ts,
        )
        run = _run_direct_persistent(payload, trace and not as_json)
        result = run["result"]

        resolved_backend = str(result.get("backend") or selected_backend)
        if selected_backend == "postgres" and resolved_backend == "memory" and not as_json:
            typer.echo("[chat] postgres backend unavailable. fell back to memory.")

        if as_json:
            typer.echo(
                json.dumps(
                    {
                        "turn": turn,
                        "thread_ts": session_thread_ts,
                        **run,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            typer.echo(f"bot [{result.get('action')}]:")
            typer.echo(str(result.get("response_text") or ""))

        turn += 1
