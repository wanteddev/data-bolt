import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from data_bolt.botctl.main import app

runner = CliRunner()


def test_simulate_requires_exactly_one_input() -> None:
    result = runner.invoke(app, ["simulate", "--text", "hello", "--case", "greeting"])
    assert result.exit_code != 0
    assert "exactly one of --text, --case, or --file" in result.output


def test_simulate_with_text_outputs_summary(monkeypatch) -> None:
    def fake_direct(payload: dict[str, Any], trace_enabled: bool) -> dict[str, Any]:
        assert payload["text"] == "안녕하세요"
        assert payload["channel_type"] == "im"
        assert trace_enabled is True
        return {
            "mode": "direct",
            "payload": payload,
            "trace": [{"node": "compose_response", "reason": "ok"}],
            "result": {
                "action": "chat_reply",
                "should_respond": True,
                "candidate_sql": None,
                "response_text": "안녕하세요! 무엇을 도와드릴까요?",
            },
        }

    monkeypatch.setattr("data_bolt.botctl.simulate._run_direct_with_trace", fake_direct)

    result = runner.invoke(app, ["simulate", "--text", "안녕하세요"])
    assert result.exit_code == 0
    assert "action: chat_reply" in result.output
    assert "response_text:" in result.output


def test_simulate_with_case_outputs_json(monkeypatch) -> None:
    def fake_direct(payload: dict[str, Any], trace_enabled: bool) -> dict[str, Any]:
        assert trace_enabled is False
        return {
            "mode": "direct",
            "payload": payload,
            "trace": [{"node": "compose_response", "reason": "ok"}],
            "result": {
                "action": "sql_generate",
                "should_respond": True,
                "candidate_sql": "SELECT 1;",
                "response_text": "sql response",
            },
        }

    monkeypatch.setattr("data_bolt.botctl.simulate._run_direct_with_trace", fake_direct)

    result = runner.invoke(app, ["simulate", "--case", "sql_gen", "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["mode"] == "direct"
    assert parsed["result"]["candidate_sql"] == "SELECT 1;"


def test_simulate_with_file_runs_multiple_cases(monkeypatch) -> None:
    def fake_direct(payload: dict[str, Any], trace_enabled: bool) -> dict[str, Any]:
        return {
            "mode": "direct",
            "payload": payload,
            "trace": [{"node": "compose_response", "reason": "ok"}],
            "result": {
                "action": "chat_reply",
                "should_respond": True,
                "candidate_sql": None,
                "response_text": payload["text"],
            },
        }

    monkeypatch.setattr("data_bolt.botctl.simulate._run_direct_with_trace", fake_direct)
    case_file = Path(__file__).parent / "fixtures" / "botctl_cases.json"

    result = runner.invoke(app, ["simulate", "--file", str(case_file)])
    assert result.exit_code == 0
    assert "[case 1]" in result.output
    assert "[case 4]" in result.output


def test_simulate_via_background_uses_background_path(monkeypatch) -> None:
    async def fake_background(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "action": "chat_reply",
            "should_respond": True,
            "candidate_sql": None,
            "response_text": f"bg:{payload['text']}",
        }

    monkeypatch.setattr("data_bolt.botctl.simulate._run_via_background", fake_background)

    result = runner.invoke(app, ["simulate", "--text", "안녕하세요", "--via-background", "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["mode"] == "background"
    assert parsed["result"]["response_text"] == "bg:안녕하세요"


def test_simulate_no_trace_disables_trace_output(monkeypatch) -> None:
    def fake_direct(payload: dict[str, Any], trace_enabled: bool) -> dict[str, Any]:
        assert trace_enabled is False
        return {
            "mode": "direct",
            "payload": payload,
            "result": {
                "action": "chat_reply",
                "should_respond": True,
                "candidate_sql": None,
                "response_text": "ok",
            },
        }

    monkeypatch.setattr("data_bolt.botctl.simulate._run_direct_with_trace", fake_direct)
    result = runner.invoke(app, ["simulate", "--text", "안녕하세요", "--no-trace"])
    assert result.exit_code == 0


def test_simulate_rejects_thread_ts_on_memory_backend(monkeypatch) -> None:
    monkeypatch.delenv("LANGGRAPH_CHECKPOINT_BACKEND", raising=False)
    result = runner.invoke(app, ["simulate", "--text", "안녕하세요", "--thread-ts", "demo-1"])
    assert result.exit_code != 0
    assert "memory backend에서" in result.output
    assert "지원되지 않습니다" in result.output


def test_simulate_allows_thread_ts_on_postgres_backend(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "postgres")

    def fake_direct(payload: dict[str, Any], trace_enabled: bool) -> dict[str, Any]:
        assert payload["thread_ts"] == "demo-1"
        return {
            "mode": "direct",
            "payload": payload,
            "result": {
                "backend": "postgres",
                "action": "chat_reply",
                "should_respond": True,
                "candidate_sql": None,
                "response_text": "ok",
            },
        }

    monkeypatch.setattr("data_bolt.botctl.simulate._run_direct_persistent", fake_direct)
    result = runner.invoke(
        app,
        ["simulate", "--text", "안녕하세요", "--thread-ts", "demo-1", "--no-trace"],
    )
    assert result.exit_code == 0


def test_simulate_allows_thread_ts_on_dynamodb_backend(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "dynamodb")

    def fake_direct(payload: dict[str, Any], trace_enabled: bool) -> dict[str, Any]:
        assert payload["thread_ts"] == "demo-1"
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

    monkeypatch.setattr("data_bolt.botctl.simulate._run_direct_persistent", fake_direct)
    result = runner.invoke(
        app,
        ["simulate", "--text", "안녕하세요", "--thread-ts", "demo-1", "--no-trace"],
    )
    assert result.exit_code == 0


def test_simulate_persistent_trace_includes_rag_and_laas(monkeypatch) -> None:
    def fake_run_single(payload: dict[str, Any], via_background: bool) -> dict[str, Any]:
        assert via_background is False
        return {
            "mode": "direct",
            "payload": payload,
            "result": {
                "action": "sql_generate",
                "should_respond": True,
                "candidate_sql": None,
                "response_text": "ok",
                "routing": {
                    "route": "data",
                    "confidence": 0.9,
                    "reason": "data request",
                    "actions": ["sql_generate"],
                },
                "generation_result": {
                    "meta": {
                        "rag": {"attempted": True, "schema_docs": 12, "glossary_docs": 3},
                        "llm": {
                            "provider": "openai_compatible",
                            "called": True,
                            "success": True,
                            "model": "glm-4.7",
                        },
                    }
                },
            },
        }

    monkeypatch.setattr("data_bolt.botctl.simulate._run_single", fake_run_single)

    result = runner.invoke(
        app,
        ["simulate", "--text", "스키마 찾아줘", "--thread-ts", "demo-1", "--json"],
        env={"LANGGRAPH_CHECKPOINT_BACKEND": "postgres"},
    )
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    nodes = [entry["node"] for entry in parsed["trace"]]
    assert "plan_turn_action" in nodes
    assert "route_decision" in nodes
    assert "sql_generate" in nodes
    assert "rag_context_lookup" in nodes
    assert "laas_completion" in nodes


def test_simulate_trace_includes_dry_run_validation_node(monkeypatch) -> None:
    def fake_run_single(payload: dict[str, Any], via_background: bool) -> dict[str, Any]:
        assert via_background is False
        return {
            "mode": "direct",
            "payload": payload,
            "result": {
                "action": "sql_generate",
                "should_respond": True,
                "candidate_sql": "SELECT 1;",
                "routing": {
                    "runtime_mode": "loop",
                    "route": "data",
                    "confidence": 0.9,
                    "reason": "data request",
                    "actions": ["sql_generate"],
                },
                "validation": {
                    "success": False,
                    "attempts": 2,
                    "error": "Syntax error",
                },
                "response_text": "dry-run failed",
                "generation_result": {
                    "meta": {
                        "llm": {
                            "provider": "openai_compatible",
                            "called": True,
                            "success": True,
                            "model": "glm-4.7",
                        },
                    }
                },
            },
        }

    monkeypatch.setattr("data_bolt.botctl.simulate._run_single", fake_run_single)
    result = runner.invoke(
        app,
        ["simulate", "--text", "쿼리 만들어줘", "--thread-ts", "demo-2", "--json"],
        env={"LANGGRAPH_CHECKPOINT_BACKEND": "postgres"},
    )

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    nodes = [entry["node"] for entry in parsed["trace"]]
    assert "validate_candidate_sql" in nodes
    assert "guarded_execute" in nodes
