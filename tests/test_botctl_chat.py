import json
from typing import Any

from typer.testing import CliRunner

from data_bolt.botctl.main import app

runner = CliRunner()


def test_chat_help_works() -> None:
    result = runner.invoke(app, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--thread-ts" in result.output


def test_chat_uses_same_thread_ts_across_turns(monkeypatch) -> None:
    seen: list[str] = []

    def fake_direct(payload: dict[str, Any], trace_enabled: bool) -> dict[str, Any]:
        assert trace_enabled is False
        seen.append(str(payload.get("thread_ts")))
        return {
            "mode": "direct",
            "payload": payload,
            "result": {
                "backend": "memory",
                "action": "chat_reply",
                "should_respond": True,
                "candidate_sql": None,
                "response_text": "ok",
            },
        }

    monkeypatch.setattr("data_bolt.botctl.chat._run_direct_persistent", fake_direct)

    result = runner.invoke(
        app,
        ["chat", "--thread-ts", "demo-1", "--no-trace"],
        input="안녕\n다음 턴\nexit\n",
    )
    assert result.exit_code == 0
    assert seen == ["demo-1", "demo-1"]


def test_chat_json_outputs_turn_results(monkeypatch) -> None:
    def fake_direct(payload: dict[str, Any], trace_enabled: bool) -> dict[str, Any]:
        assert trace_enabled is False
        return {
            "mode": "direct",
            "payload": payload,
            "trace": [{"node": "compose_response", "reason": "ok"}],
            "result": {
                "backend": "memory",
                "action": "chat_reply",
                "should_respond": True,
                "candidate_sql": None,
                "response_text": "ok",
            },
        }

    monkeypatch.setattr("data_bolt.botctl.chat._run_direct_persistent", fake_direct)
    result = runner.invoke(app, ["chat", "--json"], input="hello\nexit\n")
    assert result.exit_code == 0
    json_lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
    assert json_lines
    line = json_lines[0]
    parsed = json.loads(line)
    assert parsed["turn"] == 1
    assert parsed["result"]["response_text"] == "ok"


def test_chat_passes_accumulated_history_between_turns(monkeypatch) -> None:
    seen_histories: list[list[dict[str, str]] | None] = []

    def fake_direct(payload: dict[str, Any], trace_enabled: bool) -> dict[str, Any]:
        assert trace_enabled is False
        seen_histories.append(payload.get("history"))
        response_text = "첫 응답" if len(seen_histories) == 1 else "둘째 응답"
        return {
            "mode": "direct",
            "payload": payload,
            "result": {
                "backend": "memory",
                "action": "chat_reply",
                "should_respond": True,
                "candidate_sql": None,
                "response_text": response_text,
            },
        }

    monkeypatch.setattr("data_bolt.botctl.chat._run_direct_persistent", fake_direct)
    result = runner.invoke(app, ["chat", "--no-trace"], input="첫 질문\n둘째 질문\nexit\n")
    assert result.exit_code == 0
    assert seen_histories[0] is None
    assert seen_histories[1] == [
        {"role": "user", "content": "첫 질문"},
        {"role": "assistant", "content": "첫 응답"},
    ]


def test_chat_trims_history_with_max_turns_window(monkeypatch) -> None:
    seen_histories: list[list[dict[str, str]] | None] = []

    def fake_direct(payload: dict[str, Any], trace_enabled: bool) -> dict[str, Any]:
        assert trace_enabled is False
        seen_histories.append(payload.get("history"))
        return {
            "mode": "direct",
            "payload": payload,
            "result": {
                "backend": "memory",
                "action": "chat_reply",
                "should_respond": True,
                "candidate_sql": None,
                "response_text": f"응답{len(seen_histories)}",
            },
        }

    monkeypatch.setenv("BOTCTL_CHAT_HISTORY_MAX_TURNS", "1")
    monkeypatch.setattr("data_bolt.botctl.chat._run_direct_persistent", fake_direct)
    result = runner.invoke(app, ["chat", "--no-trace"], input="질문1\n질문2\n질문3\nexit\n")

    assert result.exit_code == 0
    assert seen_histories[0] is None
    assert seen_histories[1] == [
        {"role": "user", "content": "질문1"},
        {"role": "assistant", "content": "응답1"},
    ]
    assert seen_histories[2] == [
        {"role": "user", "content": "질문2"},
        {"role": "assistant", "content": "응답2"},
    ]


def test_chat_trace_mode_outputs_trace_line(monkeypatch) -> None:
    def fake_direct(payload: dict[str, Any], trace_enabled: bool) -> dict[str, Any]:
        if trace_enabled:
            import typer

            typer.echo("[trace] ingest: ok")
        return {
            "mode": "direct",
            "payload": payload,
            "result": {
                "backend": "memory",
                "action": "chat_reply",
                "should_respond": True,
                "candidate_sql": None,
                "response_text": "ok",
            },
        }

    monkeypatch.setattr("data_bolt.botctl.chat._run_direct_persistent", fake_direct)
    result = runner.invoke(app, ["chat", "--trace"], input="hello\nexit\n")
    assert result.exit_code == 0
    assert "[trace] ingest: ok" in result.output


def test_chat_banner_shows_dynamodb_backend(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "dynamodb")

    def fake_direct(payload: dict[str, Any], trace_enabled: bool) -> dict[str, Any]:
        assert trace_enabled is False
        return {
            "mode": "direct",
            "payload": payload,
            "result": {
                "backend": "dynamodb",
                "action": "chat_reply",
                "should_respond": True,
                "candidate_sql": None,
                "response_text": "ok",
            },
        }

    monkeypatch.setattr("data_bolt.botctl.chat._run_direct_persistent", fake_direct)
    result = runner.invoke(app, ["chat", "--no-trace"], input="hello\nexit\n")
    assert result.exit_code == 0
    assert "[chat] backend=dynamodb" in result.output
