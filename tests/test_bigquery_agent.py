import os
from uuid import uuid4

import pytest

from data_bolt.tasks.bigquery_agent import service as bigquery_agent


@pytest.fixture(autouse=True)
def _default_env(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "memory")
    monkeypatch.setenv("BIGQUERY_INTENT_LLM_ENABLED", "false")
    monkeypatch.setenv("BIGQUERY_CHAT_ALLOW_EXECUTE_IN_CHAT", "false")
    monkeypatch.delenv("LANGGRAPH_POSTGRES_FALLBACK_TO_MEMORY", raising=False)
    bigquery_agent._memory_runtime_cache.clear()
    bigquery_agent._postgres_graph_cache.clear()
    bigquery_agent._postgres_context_cache.clear()
    bigquery_agent._postgres_setup_done.clear()
    bigquery_agent._dynamodb_graph_cache.clear()


def test_run_bigquery_agent_ignores_irrelevant_channel_message() -> None:
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


def test_run_bigquery_agent_routes_data_workflow(monkeypatch) -> None:
    def fake_build(payload):
        assert payload["text"] == "전체 사용자 수 알려줘"
        return {
            "answer_structured": {
                "sql": "SELECT COUNT(*) AS cnt FROM users;",
                "explanation": "가입자 수를 집계합니다.",
            },
            "validation": {"success": True, "total_bytes_processed": 1, "estimated_cost_usd": 0.0},
        }

    monkeypatch.setattr(bigquery_agent, "build_bigquery_sql", fake_build)
    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D1",
            "thread_ts": "100.2",
            "channel_type": "im",
            "text": "전체 사용자 수 알려줘",
        }
    )
    assert result["routing"]["route"] == "data"
    assert result["candidate_sql"] == "SELECT COUNT(*) AS cnt FROM users;"
    assert "Dry-run" in result["response_text"]


def test_run_bigquery_agent_routes_free_chat_and_uses_planner(monkeypatch) -> None:
    monkeypatch.setenv("BIGQUERY_INTENT_LLM_ENABLED", "true")

    def fake_intent(**_kwargs):
        return {
            "intent": "free_chat",
            "confidence": 0.91,
            "reason": "general",
            "actions": ["none"],
        }

    def fake_plan(**_kwargs):
        return {
            "assistant_response": "안녕하세요! 무엇을 도와드릴까요?",
            "actions": ["none"],
            "action_reason": "no data task",
        }

    monkeypatch.setattr(bigquery_agent, "classify_intent_with_laas", fake_intent)
    monkeypatch.setattr(bigquery_agent, "plan_free_chat_with_laas", fake_plan)

    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D1",
            "thread_ts": "100.21",
            "channel_type": "im",
            "text": "안녕하세요",
        }
    )

    assert result["routing"]["route"] == "chat"
    assert result["intent"] == "chat"
    assert result["response_text"] == "안녕하세요! 무엇을 도와드릴까요?"


def test_run_bigquery_agent_low_confidence_falls_back_to_chat(monkeypatch) -> None:
    monkeypatch.setenv("BIGQUERY_INTENT_LLM_ENABLED", "true")
    monkeypatch.setenv("BIGQUERY_INTENT_CONFIDENCE_THRESHOLD", "0.8")

    monkeypatch.setattr(
        bigquery_agent,
        "classify_intent_with_laas",
        lambda **_kwargs: {
            "intent": "data_workflow",
            "confidence": 0.2,
            "reason": "uncertain",
            "actions": ["text_to_sql"],
        },
    )
    monkeypatch.setattr(
        bigquery_agent,
        "plan_free_chat_with_laas",
        lambda **_kwargs: {
            "assistant_response": "요청을 조금 더 구체화해 주세요.",
            "actions": ["none"],
            "action_reason": "low confidence",
        },
    )

    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D1",
            "thread_ts": "100.22",
            "channel_type": "im",
            "text": "이거 좀 봐줘",
        }
    )

    assert result["routing"]["route"] == "chat"
    assert "구체화" in result["response_text"]


def test_run_bigquery_agent_chat_planner_can_call_data_node(monkeypatch) -> None:
    monkeypatch.setenv("BIGQUERY_INTENT_LLM_ENABLED", "true")
    monkeypatch.setattr(
        bigquery_agent,
        "classify_intent_with_laas",
        lambda **_kwargs: {
            "intent": "free_chat",
            "confidence": 0.95,
            "reason": "chat then sql",
            "actions": ["none"],
        },
    )
    monkeypatch.setattr(
        bigquery_agent,
        "plan_free_chat_with_laas",
        lambda **_kwargs: {
            "assistant_response": "쿼리를 같이 만들었어요.",
            "actions": ["text_to_sql"],
            "action_reason": "needs query",
        },
    )
    monkeypatch.setattr(
        bigquery_agent,
        "build_bigquery_sql",
        lambda _payload: {
            "answer_structured": {"sql": "SELECT 1;", "explanation": "ok"},
            "validation": {"success": True, "total_bytes_processed": 0, "estimated_cost_usd": 0.0},
        },
    )

    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D1",
            "thread_ts": "100.3",
            "channel_type": "im",
            "text": "그럼 쿼리도 만들어줘",
        }
    )

    assert result["routing"]["route"] == "chat"
    assert result["candidate_sql"] == "SELECT 1;"
    assert "쿼리를 같이 만들었어요." in result["response_text"]


def test_run_bigquery_agent_does_not_use_choices_text_as_sql(monkeypatch) -> None:
    monkeypatch.setattr(
        bigquery_agent,
        "build_bigquery_sql",
        lambda _payload: {
            "answer_structured": {"sql": None, "explanation": "설명문과 SQL이 섞인 응답"},
            "choices": [{"message": {"content": "아래 SQL을 실행하세요.\nSELECT 1 AS value;"}}],
            "validation": {},
        },
    )
    monkeypatch.setattr(
        bigquery_agent,
        "dry_run_bigquery_sql",
        lambda _sql: (_ for _ in ()).throw(AssertionError("dry-run should not be called")),
    )

    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D1",
            "thread_ts": "100.31",
            "channel_type": "im",
            "text": "사용자 수 알려줘",
        }
    )

    assert result["routing"]["route"] == "data"
    assert result["candidate_sql"] is None
    assert "SELECT 1 AS value" not in result["response_text"]


def test_run_bigquery_agent_executes_when_requested(monkeypatch) -> None:
    called: dict[str, str] = {}

    def fake_dry_run(sql: str):
        called["dry_sql"] = sql
        return {"success": True, "total_bytes_processed": 0, "estimated_cost_usd": 0.0}

    def fake_execute(sql: str):
        called["exec_sql"] = sql
        return {"success": True, "job_id": "job_1", "row_count": 1, "preview_rows": [{"value": 1}]}

    monkeypatch.setattr(bigquery_agent, "dry_run_bigquery_sql", fake_dry_run)
    monkeypatch.setattr(bigquery_agent, "execute_bigquery_sql", fake_execute)

    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D2",
            "thread_ts": "100.4",
            "channel_type": "im",
            "text": "이 쿼리 실행해줘\n```sql\nSELECT 1 AS value;\n```",
        }
    )

    assert result["intent"] == "execute_sql"
    assert called["dry_sql"] == "SELECT 1 AS value;"
    assert called["exec_sql"] == "SELECT 1 AS value;"
    assert result["execution"]["success"] is True


def test_run_bigquery_agent_blocks_write_or_ddl_sql_execution(monkeypatch) -> None:
    execute_called = False

    monkeypatch.setattr(
        bigquery_agent,
        "dry_run_bigquery_sql",
        lambda _sql: {"success": True, "total_bytes_processed": 0, "estimated_cost_usd": 0.0},
    )

    def fake_execute(_sql: str):
        nonlocal execute_called
        execute_called = True
        return {"success": True}

    monkeypatch.setattr(bigquery_agent, "execute_bigquery_sql", fake_execute)

    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D2",
            "thread_ts": "100.401",
            "channel_type": "im",
            "text": "이 쿼리 실행해줘\n```sql\nDROP TABLE users;\n```",
        }
    )

    assert result["intent"] == "execute_sql"
    assert result["execution"]["success"] is False
    assert "읽기 전용 SELECT" in result["execution"]["error"]
    assert execute_called is False


def test_run_bigquery_agent_requires_sql_block_for_execution(monkeypatch) -> None:
    monkeypatch.setattr(
        bigquery_agent,
        "build_bigquery_sql",
        lambda _payload: (_ for _ in ()).throw(AssertionError("build should not be called")),
    )

    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D2",
            "thread_ts": "100.402",
            "channel_type": "im",
            "text": "지난주 가입자 수 쿼리 실행해줘",
        }
    )

    assert result["intent"] == "execute_sql"
    assert result["execution"]["success"] is False
    assert "SQL 코드 블록" in result["execution"]["error"]
    assert result["candidate_sql"] is None


def test_run_bigquery_agent_does_not_leak_previous_response_on_ignore(monkeypatch) -> None:
    monkeypatch.setattr(
        bigquery_agent,
        "build_bigquery_sql",
        lambda _payload: {
            "answer_structured": {"sql": "SELECT 1;", "explanation": "ok"},
            "validation": {"success": True, "total_bytes_processed": 0, "estimated_cost_usd": 0.0},
        },
    )

    first = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "C1",
            "thread_ts": "100.41",
            "channel_type": "im",
            "text": "데이터 조회해줘",
        }
    )
    assert first["should_respond"] is True
    assert first["response_text"]

    second = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "C1",
            "thread_ts": "100.41",
            "channel_type": "channel",
            "text": "점심 뭐먹지?",
        }
    )
    assert second["should_respond"] is False
    assert second["response_text"] == ""
    assert second["candidate_sql"] is None
    assert second["validation"] == {}
    assert second["execution"] == {}


def test_run_bigquery_agent_chat_turn_does_not_reuse_previous_sql(monkeypatch) -> None:
    monkeypatch.setenv("BIGQUERY_INTENT_LLM_ENABLED", "true")
    monkeypatch.setattr(
        bigquery_agent,
        "build_bigquery_sql",
        lambda _payload: {
            "answer_structured": {"sql": "SELECT 42 AS answer;", "explanation": "ok"},
            "validation": {"success": True, "total_bytes_processed": 0, "estimated_cost_usd": 0.0},
        },
    )

    def fake_intent(**kwargs):
        text = kwargs.get("text", "")
        if "사용자 수" in text:
            return {
                "intent": "data_workflow",
                "confidence": 0.95,
                "reason": "data task",
                "actions": ["text_to_sql"],
            }
        return {
            "intent": "free_chat",
            "confidence": 0.9,
            "reason": "chat",
            "actions": ["none"],
        }

    monkeypatch.setattr(bigquery_agent, "classify_intent_with_laas", fake_intent)
    monkeypatch.setattr(
        bigquery_agent,
        "plan_free_chat_with_laas",
        lambda **_kwargs: {
            "assistant_response": "안녕하세요",
            "actions": ["none"],
            "action_reason": "chat",
        },
    )

    first = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D5",
            "thread_ts": "100.42",
            "channel_type": "im",
            "text": "사용자 수 알려줘",
        }
    )
    assert first["candidate_sql"] == "SELECT 42 AS answer;"

    second = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D5",
            "thread_ts": "100.42",
            "channel_type": "im",
            "text": "안녕",
        }
    )
    assert second["routing"]["route"] == "chat"
    assert second["candidate_sql"] is None
    assert "SELECT 42 AS answer;" not in second["response_text"]


def test_run_bigquery_agent_ignores_message_without_appending_conversation(monkeypatch) -> None:
    monkeypatch.setattr(
        bigquery_agent,
        "build_bigquery_sql",
        lambda _payload: {
            "answer_structured": {"sql": "SELECT 1;", "explanation": "ok"},
            "validation": {"success": True, "total_bytes_processed": 0, "estimated_cost_usd": 0.0},
        },
    )

    first = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "C9",
            "thread_ts": "100.43",
            "channel_type": "channel",
            "text": "잡담",
        }
    )
    assert first["should_respond"] is False
    assert first["conversation_turns"] == 0

    second = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "C9",
            "thread_ts": "100.43",
            "channel_type": "im",
            "text": "데이터 조회",
        }
    )
    assert second["should_respond"] is True
    assert second["conversation_turns"] == 2


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


def test_run_bigquery_agent_checkpoint_error_is_fail_fast_by_default(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "postgres")

    monkeypatch.setattr(
        bigquery_agent,
        "_invoke_graph_with_postgres",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("LANGGRAPH_POSTGRES_URI is not set")
        ),
    )

    with pytest.raises(ValueError, match="LANGGRAPH_POSTGRES_URI is not set"):
        bigquery_agent.run_bigquery_agent(
            {
                "team_id": "T1",
                "channel_id": "C1",
                "thread_ts": "100.55",
                "channel_type": "im",
                "text": "테스트",
            }
        )


def test_run_bigquery_agent_fallbacks_to_memory_when_opted_in(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "postgres")
    monkeypatch.setenv("LANGGRAPH_POSTGRES_FALLBACK_TO_MEMORY", "true")

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


def test_run_bigquery_agent_dynamodb_requires_table_env(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "dynamodb")
    monkeypatch.delenv("LANGGRAPH_DYNAMODB_TABLE", raising=False)

    with pytest.raises(ValueError, match="LANGGRAPH_DYNAMODB_TABLE is not set"):
        bigquery_agent.run_bigquery_agent(
            {
                "team_id": "T1",
                "channel_id": "C1",
                "thread_ts": "100.7",
                "channel_type": "im",
                "text": "테스트",
            }
        )


def test_run_bigquery_agent_uses_dynamodb_backend(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "dynamodb")
    monkeypatch.setenv("LANGGRAPH_DYNAMODB_TABLE", "langgraph-checkpoints")

    def fake_ddb_invoke(*_args, **_kwargs):
        return {
            "should_respond": True,
            "intent": "text_to_sql",
            "response_text": "ok",
            "generation_result": {},
            "conversation": [],
        }

    monkeypatch.setattr(bigquery_agent, "_invoke_graph_with_dynamodb", fake_ddb_invoke)

    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "C1",
            "thread_ts": "100.8",
            "channel_type": "im",
            "text": "테스트",
        }
    )

    assert result["backend"] == "dynamodb"
    assert result["response_text"] == "ok"


def test_run_bigquery_agent_dynamodb_fail_fast(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "dynamodb")
    monkeypatch.setenv("LANGGRAPH_DYNAMODB_TABLE", "langgraph-checkpoints")

    def fake_ddb_invoke(*_args, **_kwargs):
        raise RuntimeError("dynamodb unavailable")

    monkeypatch.setattr(bigquery_agent, "_invoke_graph_with_dynamodb", fake_ddb_invoke)

    with pytest.raises(RuntimeError, match="dynamodb unavailable"):
        bigquery_agent.run_bigquery_agent(
            {
                "team_id": "T1",
                "channel_id": "C1",
                "thread_ts": "100.9",
                "channel_type": "im",
                "text": "테스트",
            }
        )


def test_postgres_graph_is_compiled_once_per_connection(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_POSTGRES_URI", "postgresql://user:pass@localhost:5432/db")
    compile_calls = 0
    enter_calls = 0

    class FakeCompiled:
        def invoke(self, input_state, config):
            return {**input_state, "should_respond": True}

    class FakeBuilder:
        def compile(self, checkpointer):
            nonlocal compile_calls
            assert checkpointer is not None
            compile_calls += 1
            return FakeCompiled()

    class FakeContextManager:
        def __enter__(self):
            nonlocal enter_calls
            enter_calls += 1
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(bigquery_agent, "_build_graph", lambda: FakeBuilder())
    monkeypatch.setattr(bigquery_agent, "_ensure_postgres_setup", lambda _conn: None)
    monkeypatch.setattr(
        "langgraph.checkpoint.postgres.PostgresSaver.from_conn_string",
        lambda _conn: FakeContextManager(),
    )

    state = {"text": "q", "channel_type": "im"}
    bigquery_agent._invoke_graph_with_postgres(state, "thread-1")
    bigquery_agent._invoke_graph_with_postgres(state, "thread-2")

    assert compile_calls == 1
    assert enter_calls == 1


def test_dynamodb_graph_is_compiled_once_per_configuration(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_DYNAMODB_TABLE", "langgraph-checkpoints")
    monkeypatch.setenv("LANGGRAPH_DYNAMODB_REGION", "ap-northeast-2")
    compile_calls = 0
    saver_calls = 0

    class FakeCompiled:
        def invoke(self, input_state, config):
            return {**input_state, "should_respond": True}

    class FakeBuilder:
        def compile(self, checkpointer):
            nonlocal compile_calls
            assert checkpointer == "fake-saver"
            compile_calls += 1
            return FakeCompiled()

    def fake_build_saver(*, table_name: str, region_name: str | None, endpoint_url: str | None):
        nonlocal saver_calls
        saver_calls += 1
        assert table_name == "langgraph-checkpoints"
        assert region_name == "ap-northeast-2"
        assert endpoint_url is None
        return "fake-saver"

    monkeypatch.setattr(bigquery_agent, "_build_graph", lambda: FakeBuilder())
    monkeypatch.setattr(bigquery_agent, "_build_dynamodb_saver", fake_build_saver)

    state = {"text": "q", "channel_type": "im"}
    bigquery_agent._invoke_graph_with_dynamodb(state, "thread-1")
    bigquery_agent._invoke_graph_with_dynamodb(state, "thread-2")

    assert compile_calls == 1
    assert saver_calls == 1


@pytest.mark.integration
def test_run_bigquery_agent_persists_state_with_postgres(monkeypatch) -> None:
    postgres_uri = os.getenv("TEST_LANGGRAPH_POSTGRES_URI")
    if not postgres_uri:
        pytest.skip("TEST_LANGGRAPH_POSTGRES_URI is not set")

    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "postgres")
    monkeypatch.setenv("LANGGRAPH_POSTGRES_URI", postgres_uri)
    monkeypatch.setenv("BIGQUERY_INTENT_LLM_ENABLED", "false")
    monkeypatch.setattr(
        bigquery_agent,
        "build_bigquery_sql",
        lambda _payload: {
            "answer_structured": {"sql": "SELECT 1;", "explanation": "ok"},
            "validation": {"success": True, "total_bytes_processed": 0, "estimated_cost_usd": 0.0},
        },
    )

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
