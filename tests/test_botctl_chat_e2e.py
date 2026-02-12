from typer.testing import CliRunner

from data_bolt.botctl.main import app
from data_bolt.tasks.bigquery_agent import service as bigquery_agent

runner = CliRunner()


def _fake_plan_turn_action(**kwargs):
    text = str(kwargs.get("text") or "")
    has_user_sql_block = bool(kwargs.get("has_user_sql_block"))
    has_last_candidate_sql = bool(kwargs.get("has_last_candidate_sql"))
    pending_execution_sql = bool(kwargs.get("pending_execution_sql"))

    if "실행 취소" in text:
        return {"action": "execution_cancel", "confidence": 0.95, "reason": "cancel requested"}
    if "실행 승인" in text:
        return {"action": "execution_approve", "confidence": 0.95, "reason": "approve requested"}
    if "실행" in text and (
        "쿼리" in text or "SQL" in text or has_user_sql_block or has_last_candidate_sql
    ):
        return {"action": "sql_execute", "confidence": 0.95, "reason": "execution requested"}
    if pending_execution_sql and ("쿼리" in text or "sql" in text.lower()):
        return {"action": "sql_execute", "confidence": 0.9, "reason": "pending sql referenced"}
    if any(
        keyword in text for keyword in ["사용자", "가입", "조회", "집계", "지표", "추이", "데이터"]
    ):
        return {"action": "sql_generate", "confidence": 0.9, "reason": "data request"}
    return {"action": "chat_reply", "confidence": 0.8, "reason": "general chat"}


def _fake_sql_build(_payload):
    return {
        "answer_structured": {
            "sql": "SELECT COUNT(*) AS total_users FROM users;",
            "explanation": "전체 사용자 수를 집계합니다.",
        },
        "validation": {
            "success": True,
            "sql": "SELECT COUNT(*) AS total_users FROM users;",
            "total_bytes_processed": 42,
            "estimated_cost_usd": 0.0,
        },
    }


def _reset_memory_state() -> None:
    bigquery_agent._memory_runtime_cache.clear()
    bigquery_agent._postgres_graph_cache.clear()
    bigquery_agent._postgres_context_cache.clear()
    bigquery_agent._postgres_setup_done.clear()
    bigquery_agent._dynamodb_graph_cache.clear()
    bigquery_agent._loop_memory_runtime_cache.clear()
    bigquery_agent._loop_postgres_graph_cache.clear()
    bigquery_agent._loop_postgres_context_cache.clear()
    bigquery_agent._loop_postgres_setup_done.clear()
    bigquery_agent._loop_dynamodb_graph_cache.clear()


def test_chat_e2e_generate_auto_executes_when_cost_is_low(monkeypatch) -> None:
    _reset_memory_state()
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "memory")
    monkeypatch.setenv("BIGQUERY_AGENT_RUNTIME_MODE", "graph")
    monkeypatch.setenv("BIGQUERY_INTENT_LLM_ENABLED", "true")
    monkeypatch.setenv("BIGQUERY_ACTION_ROUTER_LLM_ENABLED", "true")
    monkeypatch.setenv("BIGQUERY_CHAT_PLANNER_ENABLED", "false")
    monkeypatch.setenv("BIGQUERY_RESULT_INSIGHT_LLM_ENABLED", "false")
    monkeypatch.setenv("BIGQUERY_AUTO_EXECUTE_MAX_COST_USD", "1.0")
    monkeypatch.setattr(bigquery_agent, "plan_turn_action", _fake_plan_turn_action)
    monkeypatch.setattr(bigquery_agent, "build_bigquery_sql", _fake_sql_build)

    executed: dict[str, str] = {}

    def fake_execute(sql: str):
        executed["sql"] = sql
        return {"success": True, "job_id": "job_e2e_1", "row_count": 1, "preview_rows": [{"v": 1}]}

    monkeypatch.setattr(bigquery_agent, "execute_bigquery_sql", fake_execute)

    result = runner.invoke(
        app,
        ["chat", "--thread-ts", "e2e-exec-1", "--no-trace"],
        input="전체 사용자 수를 구하고 싶어요.\nexit\n",
    )

    assert result.exit_code == 0
    assert ":rocket: 쿼리 실행 완료" in result.output
    assert executed["sql"] == "SELECT COUNT(*) AS total_users FROM users;"


def test_chat_e2e_cancel_clears_pending(monkeypatch) -> None:
    _reset_memory_state()
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "memory")
    monkeypatch.setenv("BIGQUERY_AGENT_RUNTIME_MODE", "graph")
    monkeypatch.setenv("BIGQUERY_INTENT_LLM_ENABLED", "true")
    monkeypatch.setenv("BIGQUERY_ACTION_ROUTER_LLM_ENABLED", "true")
    monkeypatch.setenv("BIGQUERY_CHAT_PLANNER_ENABLED", "false")
    monkeypatch.setenv("BIGQUERY_RESULT_INSIGHT_LLM_ENABLED", "false")
    monkeypatch.setenv("BIGQUERY_AUTO_EXECUTE_MAX_COST_USD", "0")
    monkeypatch.setattr(bigquery_agent, "plan_turn_action", _fake_plan_turn_action)
    monkeypatch.setattr(bigquery_agent, "build_bigquery_sql", _fake_sql_build)
    monkeypatch.setattr(
        bigquery_agent,
        "execute_bigquery_sql",
        lambda _sql: (_ for _ in ()).throw(AssertionError("execute should not be called")),
    )

    result = runner.invoke(
        app,
        ["chat", "--thread-ts", "e2e-exec-2", "--no-trace"],
        input="전체 사용자 수 알려줘\n이 쿼리 실행해줘\n실행 취소\n실행 승인\nexit\n",
    )

    assert result.exit_code == 0
    assert "쿼리 실행 요청을 취소했습니다" in result.output
    assert "승인할 대기 중인 쿼리가 없습니다" in result.output


def test_chat_e2e_previous_turn_sql_uses_approval_flow(monkeypatch) -> None:
    _reset_memory_state()
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "memory")
    monkeypatch.setenv("BIGQUERY_AGENT_RUNTIME_MODE", "graph")
    monkeypatch.setenv("BIGQUERY_INTENT_LLM_ENABLED", "true")
    monkeypatch.setenv("BIGQUERY_ACTION_ROUTER_LLM_ENABLED", "true")
    monkeypatch.setenv("BIGQUERY_CHAT_PLANNER_ENABLED", "false")
    monkeypatch.setenv("BIGQUERY_RESULT_INSIGHT_LLM_ENABLED", "false")
    monkeypatch.setenv("BIGQUERY_AUTO_EXECUTE_MAX_COST_USD", "0")
    monkeypatch.setattr(bigquery_agent, "plan_turn_action", _fake_plan_turn_action)
    monkeypatch.setattr(bigquery_agent, "build_bigquery_sql", _fake_sql_build)

    executed: dict[str, str] = {}

    def fake_execute(sql: str):
        executed["sql"] = sql
        return {"success": True, "job_id": "job_e2e_3", "row_count": 1, "preview_rows": [{"v": 1}]}

    monkeypatch.setattr(bigquery_agent, "execute_bigquery_sql", fake_execute)

    result = runner.invoke(
        app,
        ["chat", "--thread-ts", "e2e-exec-3", "--no-trace"],
        input="전체 사용자 수 알려줘\n방금 만든 쿼리 실행해줘\n실행 승인\nexit\n",
    )

    assert result.exit_code == 0
    assert "재확인" not in result.output
    assert "자동 실행 임계값 이상" in result.output
    assert ":rocket: 쿼리 실행 완료" in result.output
    assert executed["sql"] == "SELECT COUNT(*) AS total_users FROM users;"


def test_chat_e2e_chat_turn_does_not_leak_sql(monkeypatch) -> None:
    _reset_memory_state()
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "memory")
    monkeypatch.setenv("BIGQUERY_AGENT_RUNTIME_MODE", "graph")
    monkeypatch.setenv("BIGQUERY_INTENT_LLM_ENABLED", "true")
    monkeypatch.setenv("BIGQUERY_ACTION_ROUTER_LLM_ENABLED", "true")
    monkeypatch.setenv("BIGQUERY_CHAT_PLANNER_ENABLED", "false")
    monkeypatch.setenv("BIGQUERY_RESULT_INSIGHT_LLM_ENABLED", "false")
    monkeypatch.setenv("BIGQUERY_AUTO_EXECUTE_MAX_COST_USD", "0")
    monkeypatch.setattr(bigquery_agent, "plan_turn_action", _fake_plan_turn_action)
    monkeypatch.setattr(bigquery_agent, "build_bigquery_sql", _fake_sql_build)
    monkeypatch.setattr(
        bigquery_agent,
        "execute_bigquery_sql",
        lambda _sql: (_ for _ in ()).throw(AssertionError("execute should not be called")),
    )

    result = runner.invoke(
        app,
        ["chat", "--thread-ts", "e2e-chat-1", "--no-trace"],
        input="전체 사용자 수 알려줘\n안녕하세요\nexit\n",
    )

    assert result.exit_code == 0
    assert result.output.count("```sql") == 1
    assert "질문 의도를 정확히 파악하지 못했습니다." in result.output
