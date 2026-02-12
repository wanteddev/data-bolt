"""Simulation command for botctl."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from dotenv import load_dotenv

from data_bolt.botctl.runtime import guard_thread_ts_with_backend, resolve_checkpoint_backend
from data_bolt.botctl.types import SimulationCase, SimulationRun
from data_bolt.tasks.bigquery_agent import AgentPayload, AgentState, run_bigquery_agent

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

    direct_result = run_bigquery_agent(cast(AgentPayload, payload))
    return {"mode": "direct", "payload": payload, "result": cast(dict[str, Any], direct_result)}


def _trace_reason(node: str, state: AgentState) -> str:
    if node == "ingest":
        text = state.get("text", "")
        return f"입력 텍스트/히스토리를 conversation으로 정규화했습니다. text='{text[:40]}'"
    if node == "decide_turn":
        should_respond = bool(state.get("should_respond"))
        action = state.get("action") or "-"
        confidence = state.get("action_confidence")
        reason = state.get("action_reason") or "-"
        if not should_respond:
            return "응답 조건 미충족으로 흐름을 종료합니다."
        return (
            f"요청 액션을 '{state.get('action')}'로 선택했습니다. "
            f"confidence={confidence if confidence is not None else '-'}, reason={reason}"
        )
    if node == "route_decision":
        return f"라우팅 경로를 '{state.get('route')}'로 결정했습니다."
    if node in {"sql_generate", "sql_execute", "sql_validate_explain", "schema_lookup"}:
        action = state.get("action")
        generation_result_raw = state.get("generation_result")
        generation_result = generation_result_raw if isinstance(generation_result_raw, dict) else {}
        meta_raw = generation_result.get("meta")
        meta = meta_raw if isinstance(meta_raw, dict) else {}
        rag_raw = meta.get("rag")
        rag = rag_raw if isinstance(rag_raw, dict) else {}
        if rag.get("attempted"):
            return (
                "RAG 스키마/용어집 검색 후 응답 생성을 수행했습니다. "
                f"schema_docs={rag.get('schema_docs')}, glossary_docs={rag.get('glossary_docs')}"
            )
        if action == "sql_validate_explain" and state.get("candidate_sql"):
            return "요청 내 SQL을 사용해 dry-run/검증 경로를 수행했습니다."
        if action == "schema_lookup":
            return "스키마/RAG 기반 설명 및 참고 SQL 생성을 수행했습니다."
        if action == "sql_execute":
            return "실행 대상 SQL을 준비했습니다."
        return "sql 생성 경로를 수행했습니다."
    if node == "validate_candidate_sql":
        dry_run = state.get("dry_run") or {}
        if not state.get("candidate_sql"):
            return "검증할 SQL이 없어 dry-run 단계를 건너뜁니다."
        if dry_run.get("success"):
            return (
                "BigQuery dry-run 검증을 통과했습니다. "
                f"bytes={dry_run.get('total_bytes_processed')}, cost={dry_run.get('estimated_cost_usd')}"
            )
        return f"BigQuery dry-run 검증이 실패했습니다. error={dry_run.get('error') or '-'}"
    if node == "execute_sql":
        execute_result: dict[str, Any] = state.get("execution") or {}
        success = execute_result.get("success")
        return (
            "쿼리 실행을 완료했습니다."
            if success
            else "쿼리 실행을 시도했으나 실패/생략되었습니다."
        )
    if node == "compose_response":
        return "최종 Slack 응답 텍스트를 조합했습니다."
    return "노드 실행 완료"


def _trace_reason_from_result(result: dict[str, Any]) -> list[dict[str, str]]:
    trace: list[dict[str, str]] = [
        {"node": "ingest", "reason": "입력 텍스트를 수집했습니다."},
    ]

    if not result.get("should_respond"):
        trace.append({"node": "decide_turn", "reason": "응답 조건 미충족으로 종료했습니다."})
        trace.append({"node": "end", "reason": "응답 조건 미충족으로 종료했습니다."})
        return trace

    routing_raw = result.get("routing")
    routing = routing_raw if isinstance(routing_raw, dict) else {}
    action = str(result.get("action") or "unknown")
    confidence = routing.get("confidence")
    reason = routing.get("reason")
    route = routing.get("route")
    trace.append(
        {
            "node": "decide_turn",
            "reason": (
                f"요청 액션을 '{action}'로 선택했습니다. "
                f"confidence={confidence if confidence is not None else '-'}, "
                f"reason={reason or '-'}"
            ),
        }
    )
    trace.append(
        {"node": "route_decision", "reason": f"라우팅 경로를 '{route or '-'}'로 결정했습니다."}
    )

    generation_result_raw = result.get("generation_result")
    generation_result = generation_result_raw if isinstance(generation_result_raw, dict) else {}
    meta_raw = generation_result.get("meta")
    meta = meta_raw if isinstance(meta_raw, dict) else {}
    rag_raw = meta.get("rag")
    rag = rag_raw if isinstance(rag_raw, dict) else {}
    if rag.get("attempted"):
        schema_docs = rag.get("schema_docs")
        glossary_docs = rag.get("glossary_docs")
        trace.append(
            {
                "node": "rag_context_lookup",
                "reason": (
                    f"RAG 스키마/용어집 검색을 수행했습니다. "
                    f"schema_docs={schema_docs}, glossary_docs={glossary_docs}"
                ),
            }
        )

    if route == "data" or result.get("candidate_sql"):
        action_node = (
            action
            if action in {"schema_lookup", "sql_validate_explain", "sql_generate", "sql_execute"}
            else "chat_reply"
        )
        trace.append({"node": action_node, "reason": "액션 노드를 수행했습니다."})
    llm_raw = meta.get("llm")
    llm = llm_raw if isinstance(llm_raw, dict) else {}
    if not llm:
        laas_raw = meta.get("laas")
        llm = laas_raw if isinstance(laas_raw, dict) else {}
    if llm.get("called"):
        llm_success = bool(llm.get("success"))
        provider = llm.get("provider") or "laas"
        if llm_success:
            model = llm.get("model") or "-"
            trace.append(
                {
                    "node": "laas_completion",
                    "reason": f"LLM 호출이 성공했습니다. provider={provider}, model={model}",
                }
            )
        else:
            trace.append(
                {
                    "node": "laas_completion",
                    "reason": (
                        f"LLM 호출이 실패했습니다. provider={provider}, "
                        f"error={llm.get('error') or '-'}"
                    ),
                }
            )
    validation_raw = result.get("validation")
    validation = validation_raw if isinstance(validation_raw, dict) else {}
    if result.get("candidate_sql"):
        if validation.get("success"):
            trace.append(
                {
                    "node": "validate_candidate_sql",
                    "reason": (
                        "BigQuery dry-run 검증을 통과했습니다. "
                        f"bytes={validation.get('total_bytes_processed')}, "
                        f"cost={validation.get('estimated_cost_usd')}"
                    ),
                }
            )
        else:
            attempts = validation.get("attempts")
            trace.append(
                {
                    "node": "validate_candidate_sql",
                    "reason": (
                        "BigQuery dry-run 검증이 실패했습니다. "
                        f"attempts={attempts if attempts is not None else '-'}, "
                        f"error={validation.get('error') or '-'}"
                    ),
                }
            )
    executable_actions = {"sql_execute", "execution_approve", "execution_cancel"}
    if action in executable_actions:
        trace.append({"node": "guarded_execute", "reason": "실행 정책 점검을 수행했습니다."})
        execution_raw = result.get("execution")
        execution: dict[str, Any] = execution_raw if isinstance(execution_raw, dict) else {}
        if execution.get("success"):
            trace.append({"node": "execute_sql", "reason": "쿼리 실행을 완료했습니다."})
        else:
            trace.append(
                {"node": "execute_sql", "reason": "쿼리 실행을 시도했으나 실패/생략되었습니다."}
            )
    trace.append({"node": "compose_response", "reason": "최종 응답 텍스트를 조합했습니다."})
    return trace


def _run_direct_persistent(payload: dict[str, Any], trace_enabled: bool) -> SimulationRun:
    run = _run_single(payload, False)
    trace = _trace_reason_from_result(run["result"])
    run["trace"] = trace
    if trace_enabled:
        for entry in trace:
            typer.echo(f"[trace] {entry['node']}: {entry['reason']}")
    return run


def _build_result_from_state(state: AgentState, backend: str, thread_id: str) -> dict[str, Any]:
    generation_result = state.get("generation_result") or {}
    response_text = state.get("response_text") or ""
    validation = state.get("dry_run") or generation_result.get("validation") or {}

    return {
        "thread_id": thread_id,
        "backend": backend,
        "action": state.get("action"),
        "should_respond": bool(state.get("should_respond")),
        "response_text": response_text,
        "candidate_sql": state.get("candidate_sql"),
        "validation": validation,
        "execution": state.get("execution") or {},
        "generation_result": generation_result,
        "conversation_turns": len(state.get("conversation") or []),
        "routing": {
            "action": state.get("action"),
            "confidence": state.get("action_confidence"),
            "reason": state.get("action_reason"),
            "route": state.get("route"),
            "fallback_used": bool(state.get("fallback_used")),
        },
    }


def _run_direct_with_trace(payload: dict[str, Any], trace_enabled: bool) -> SimulationRun:
    run = _run_single(payload, False)
    trace = _trace_reason_from_result(run["result"])
    run["trace"] = trace
    if trace_enabled:
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
        typer.Option("--trace/--no-trace", help="Show live graph node trace while running."),
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
