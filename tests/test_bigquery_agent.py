import os
from uuid import uuid4

import pytest

from data_bolt.tasks import bigquery_agent, bigquery_sql


def _make_laas_build_adapter(
    raw_response: dict[str, object],
    *,
    instruction_type: str = "bigquery_sql_generation",
    validation: dict[str, object] | None = None,
):
    """Bridge LAAS-like raw responses to bigquery_agent expected shape for tests."""

    def _build(_payload):
        result = bigquery_sql.adapt_laas_response_for_agent(raw_response, instruction_type)
        if validation is not None:
            result["validation"] = validation
        return result

    return _build


def test_run_bigquery_agent_ignores_irrelevant_channel_message(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "memory")
    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "C1",
            "thread_ts": "100.1",
            "channel_type": "channel",
            "text": "오늘 점심 뭐 먹지?",
        }
    )

    assert result["should_respond"] is False
    assert result["response_text"] == ""


def test_run_bigquery_agent_generates_sql_response(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "memory")

    def fake_build(payload):
        assert payload["text"] == "가입자 수 쿼리 만들어줘"
        return {
            "answer_structured": {
                "sql": "SELECT COUNT(*) AS cnt FROM users;",
                "explanation": "가입자 수를 집계합니다.",
            },
            "validation": {
                "success": True,
                "total_bytes_processed": 1024,
                "estimated_cost_usd": 0.0,
            },
        }

    monkeypatch.setattr(bigquery_agent, "build_bigquery_sql", fake_build)

    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D1",
            "thread_ts": "100.2",
            "channel_type": "im",
            "text": "가입자 수 쿼리 만들어줘",
        }
    )

    assert result["should_respond"] is True
    assert "SELECT COUNT(*) AS cnt FROM users;" in result["response_text"]
    assert "Dry-run" in result["response_text"]


def test_run_bigquery_agent_executes_when_requested(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "memory")

    called: dict[str, str] = {}

    def fake_dry_run(sql: str):
        called["dry_sql"] = sql
        return {
            "success": True,
            "total_bytes_processed": 2048,
            "estimated_cost_usd": 0.0,
        }

    def fake_execute(sql: str):
        called["exec_sql"] = sql
        return {
            "success": True,
            "job_id": "job_1",
            "row_count": 2,
            "preview_rows": [{"value": 1}, {"value": 2}],
        }

    monkeypatch.setattr(bigquery_agent, "dry_run_bigquery_sql", fake_dry_run)
    monkeypatch.setattr(bigquery_agent, "execute_bigquery_sql", fake_execute)

    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D2",
            "thread_ts": "100.3",
            "channel_type": "im",
            "text": "이 쿼리 실행해줘\n```sql\nSELECT 1 AS value;\n```",
        }
    )

    assert result["intent"] == "execute_sql"
    assert called["dry_sql"] == "SELECT 1 AS value;"
    assert called["exec_sql"] == "SELECT 1 AS value;"
    assert result["execution"]["success"] is True


def test_run_bigquery_agent_handles_laas_response_blocks(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "memory")

    raw_response = {
        "choices": [
            {
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "{"
                                '"sql":"SELECT COUNT(*) AS cnt FROM users;",'
                                '"explanation":"LAAS 응답 포맷 테스트",'
                                '"assumptions":"",'
                                '"validation_steps":[]'
                                "}"
                            ),
                        }
                    ]
                }
            }
        ]
    }
    fake_build = _make_laas_build_adapter(
        raw_response,
        validation={
            "success": True,
            "total_bytes_processed": 1024,
            "estimated_cost_usd": 0.0,
        },
    )
    monkeypatch.setattr(bigquery_agent, "build_bigquery_sql", fake_build)

    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D3",
            "thread_ts": "100.4",
            "channel_type": "im",
            "text": "가입자 수 쿼리 만들어줘",
        }
    )

    assert result["should_respond"] is True
    assert result["candidate_sql"] == "SELECT COUNT(*) AS cnt FROM users;"
    assert "LAAS 응답 포맷 테스트" in result["response_text"]
    assert "Dry-run" in result["response_text"]


def test_run_bigquery_agent_fallbacks_only_for_checkpoint_errors(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "postgres")

    def fake_pg_invoke(*_args, **_kwargs):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(bigquery_agent, "_invoke_graph_with_postgres", fake_pg_invoke)

    with pytest.raises(RuntimeError, match="generation failed"):
        bigquery_agent.run_bigquery_agent(
            {
                "team_id": "T1",
                "channel_id": "C1",
                "thread_ts": "100.5",
                "channel_type": "im",
                "text": "테스트",
            }
        )


def test_run_bigquery_agent_fallbacks_to_memory_for_checkpoint_error(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "postgres")

    def fake_pg_invoke(*_args, **_kwargs):
        raise ValueError("LANGGRAPH_POSTGRES_URI is not set")

    def fake_mem_invoke(*_args, **_kwargs):
        return {
            "should_respond": True,
            "intent": "text_to_sql",
            "response_text": "ok",
            "generation_result": {},
            "conversation": [],
        }

    monkeypatch.setattr(bigquery_agent, "_invoke_graph_with_postgres", fake_pg_invoke)
    monkeypatch.setattr(bigquery_agent, "_invoke_graph_with_memory", fake_mem_invoke)

    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "C1",
            "thread_ts": "100.6",
            "channel_type": "im",
            "text": "테스트",
        }
    )

    assert result["backend"] == "memory"
    assert result["response_text"] == "ok"


def test_run_bigquery_agent_thread_id_defaults_when_payload_has_empty_strings(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "memory")

    def fake_build(_payload):
        return {
            "answer_structured": {
                "sql": "SELECT 1;",
                "explanation": "ok",
            },
            "validation": {"success": True},
        }

    monkeypatch.setattr(bigquery_agent, "build_bigquery_sql", fake_build)
    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "",
            "channel_id": "C-empty",
            "thread_ts": "200.2",
            "channel_type": "im",
            "text": "sql 만들어줘",
        }
    )

    assert result["thread_id"].startswith("local:C-empty:")


@pytest.mark.integration
def test_run_bigquery_agent_persists_state_with_postgres(monkeypatch) -> None:
    postgres_uri = os.getenv("TEST_LANGGRAPH_POSTGRES_URI")
    if not postgres_uri:
        pytest.skip("TEST_LANGGRAPH_POSTGRES_URI is not set")

    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "postgres")
    monkeypatch.setenv("LANGGRAPH_POSTGRES_URI", postgres_uri)

    def fake_build(_payload):
        return {
            "answer_structured": {
                "sql": "SELECT 1;",
                "explanation": "ok",
            },
            "validation": {
                "success": True,
                "total_bytes_processed": 10,
                "estimated_cost_usd": 0.0,
            },
        }

    monkeypatch.setattr(bigquery_agent, "build_bigquery_sql", fake_build)

    thread_id = f"T1:C1:{uuid4()}"
    first = bigquery_agent.run_bigquery_agent(
        {
            "thread_id": thread_id,
            "team_id": "T1",
            "channel_id": "C1",
            "thread_ts": "200.1",
            "channel_type": "im",
            "text": "첫 질문",
        }
    )
    second = bigquery_agent.run_bigquery_agent(
        {
            "thread_id": thread_id,
            "team_id": "T1",
            "channel_id": "C1",
            "thread_ts": "200.1",
            "channel_type": "im",
            "text": "두 번째 질문",
        }
    )

    assert second["conversation_turns"] > first["conversation_turns"]
