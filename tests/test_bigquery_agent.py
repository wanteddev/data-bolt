import os
from uuid import uuid4

import pytest

from data_bolt.tasks.bigquery_agent import service as bigquery_agent


def _default_plan_turn_action(**kwargs):
    text = str(kwargs.get("text") or "").strip()
    normalized = text.lower()
    has_user_sql_block = bool(kwargs.get("has_user_sql_block"))
    has_last_candidate_sql = bool(kwargs.get("has_last_candidate_sql"))
    pending_execution_sql = bool(kwargs.get("pending_execution_sql"))

    if "실행 취소" in text:
        return {"action": "execution_cancel", "confidence": 0.95, "reason": "cancel requested"}
    if "실행 승인" in text:
        return {"action": "execution_approve", "confidence": 0.95, "reason": "approve requested"}
    if "검증" in text or ("설명" in text and ("쿼리" in text or "sql" in normalized)):
        return {
            "action": "sql_validate_explain",
            "confidence": 0.9,
            "reason": "validation requested",
        }
    if "스키마" in text or "테이블" in text or "컬럼" in text:
        return {"action": "schema_lookup", "confidence": 0.9, "reason": "schema requested"}
    if (
        "실행" in text
        and ("쿼리" in text or "sql" in normalized or has_user_sql_block or has_last_candidate_sql)
    ) or pending_execution_sql:
        return {"action": "sql_execute", "confidence": 0.95, "reason": "execution requested"}
    if has_user_sql_block:
        return {"action": "sql_execute", "confidence": 0.9, "reason": "sql block provided"}
    if any(
        keyword in text for keyword in ["사용자", "가입", "조회", "집계", "지표", "추이", "데이터"]
    ):
        return {"action": "sql_generate", "confidence": 0.9, "reason": "data request"}
    return {"action": "chat_reply", "confidence": 0.8, "reason": "general chat"}


@pytest.fixture(autouse=True)
def _default_env(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_BACKEND", "memory")
    monkeypatch.setenv("BIGQUERY_INTENT_LLM_ENABLED", "true")
    monkeypatch.setenv("BIGQUERY_ACTION_ROUTER_LLM_ENABLED", "true")
    monkeypatch.setenv("BIGQUERY_CHAT_ALLOW_EXECUTE_IN_CHAT", "false")
    monkeypatch.setenv("BIGQUERY_CHAT_PLANNER_ENABLED", "false")
    monkeypatch.setenv("BIGQUERY_AUTO_EXECUTE_MAX_COST_USD", "0")
    monkeypatch.setenv("BIGQUERY_RESULT_INSIGHT_LLM_ENABLED", "false")
    monkeypatch.delenv("LANGGRAPH_POSTGRES_FALLBACK_TO_MEMORY", raising=False)
    monkeypatch.setattr(bigquery_agent, "plan_turn_action", _default_plan_turn_action)
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


def test_run_bigquery_agent_routes_free_chat_with_planner(monkeypatch) -> None:
    monkeypatch.setenv("BIGQUERY_INTENT_LLM_ENABLED", "true")
    monkeypatch.setenv("BIGQUERY_CHAT_PLANNER_ENABLED", "true")

    def fake_intent(**_kwargs):
        return {
            "action": "chat_reply",
            "confidence": 0.91,
            "reason": "general",
        }

    monkeypatch.setattr(bigquery_agent, "plan_turn_action", fake_intent)
    monkeypatch.setattr(
        bigquery_agent,
        "plan_free_chat",
        lambda **_kwargs: {
            "assistant_response": "안녕하세요. 원하는 지표와 기간을 알려주시면 바로 도와드릴게요.",
        },
    )

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
    assert result["action"] == "chat_reply"
    assert "원하는 지표와 기간" in result["response_text"]


def test_run_bigquery_agent_router_error_falls_back_to_safe_chat(monkeypatch) -> None:
    monkeypatch.setenv("BIGQUERY_INTENT_LLM_ENABLED", "true")
    monkeypatch.setenv("BIGQUERY_CHAT_PLANNER_ENABLED", "true")

    monkeypatch.setattr(
        bigquery_agent,
        "plan_turn_action",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("router unavailable")),
    )
    monkeypatch.setattr(
        bigquery_agent,
        "plan_free_chat",
        lambda **_kwargs: {
            "assistant_response": "요청을 더 구체화해주시면 바로 SQL로 바꿔드릴 수 있어요.",
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
    assert result["routing"]["fallback_used"] is True
    assert "더 구체화" in result["response_text"]


def test_run_bigquery_agent_llm_data_action_routes_to_data_node(monkeypatch) -> None:
    monkeypatch.setenv("BIGQUERY_INTENT_LLM_ENABLED", "true")
    monkeypatch.setattr(
        bigquery_agent,
        "plan_turn_action",
        lambda **_kwargs: {
            "action": "sql_generate",
            "confidence": 0.95,
            "reason": "data task",
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

    assert result["routing"]["route"] == "data"
    assert result["candidate_sql"] == "SELECT 1;"


def test_run_bigquery_agent_schema_lookup_never_executes(monkeypatch) -> None:
    monkeypatch.setattr(
        bigquery_agent,
        "plan_turn_action",
        lambda **_kwargs: {
            "action": "schema_lookup",
            "confidence": 0.95,
            "reason": "schema request",
        },
    )
    monkeypatch.setattr(
        bigquery_agent,
        "explain_schema_lookup",
        lambda **_kwargs: {
            "response_text": "signup_user 테이블에서 create_date/user_cnt를 조회할 수 있습니다.",
            "reference_sql": "SELECT create_date, user_cnt FROM wanted_stats.signup_user;",
            "meta": {},
        },
    )
    monkeypatch.setattr(
        bigquery_agent,
        "execute_bigquery_sql",
        lambda _sql: (_ for _ in ()).throw(AssertionError("execute should not be called")),
    )

    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D1",
            "thread_ts": "100.301",
            "channel_type": "im",
            "text": "어떤 테이블에서 가입자 조회가 가능해?",
        }
    )

    assert result["action"] == "schema_lookup"
    assert result["execution"] == {}
    assert "signup_user" in result["response_text"]


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

    first = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D2",
            "thread_ts": "100.4",
            "channel_type": "im",
            "text": "이 쿼리 실행해줘\n```sql\nSELECT 1 AS value;\n```",
        }
    )

    assert first["action"] == "sql_execute"
    assert called["dry_sql"] == "SELECT 1 AS value;"
    assert first["execution"]["success"] is False
    assert "실행 승인" in first["execution"]["error"]
    assert "exec_sql" not in called

    second = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D2",
            "thread_ts": "100.4",
            "channel_type": "im",
            "text": "실행 승인",
        }
    )

    assert second["action"] == "execution_approve"
    assert second["execution"]["success"] is True
    assert called["exec_sql"] == "SELECT 1 AS value;"


def test_run_bigquery_agent_auto_executes_generated_sql_when_low_cost(monkeypatch) -> None:
    monkeypatch.setenv("BIGQUERY_AUTO_EXECUTE_MAX_COST_USD", "1.0")
    executed: dict[str, str] = {}

    monkeypatch.setattr(
        bigquery_agent,
        "build_bigquery_sql",
        lambda _payload: {
            "answer_structured": {"sql": "SELECT 1 AS value;", "explanation": "ok"},
            "validation": {
                "success": True,
                "sql": "SELECT 1 AS value;",
                "total_bytes_processed": 42,
                "estimated_cost_usd": 0.2,
            },
        },
    )
    monkeypatch.setattr(
        bigquery_agent,
        "execute_bigquery_sql",
        lambda sql: (
            executed.setdefault("sql", sql)
            and {
                "success": True,
                "job_id": "job_auto_1",
                "row_count": 1,
                "preview_rows": [{"value": 1}],
            }
        ),
    )

    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D2",
            "thread_ts": "100.401",
            "channel_type": "im",
            "text": "전체 사용자 수 알려줘",
        }
    )

    assert result["action"] == "sql_generate"
    assert result["execution"]["success"] is True
    assert executed["sql"] == "SELECT 1 AS value;"
    assert result["routing"]["execution_policy"] == "auto_execute"
    assert result["routing"]["execution_policy_reason"] == "cost_below_threshold"
    assert result["routing"]["cost_threshold_usd"] == 1.0
    assert result["routing"]["estimated_cost_usd"] == 0.2
    assert ":bar_chart: 실행 결과 미리보기" in result["response_text"]
    assert "preview: `" not in result["response_text"]


def test_run_bigquery_agent_adds_llm_result_insight_sections(monkeypatch) -> None:
    monkeypatch.setenv("BIGQUERY_RESULT_INSIGHT_LLM_ENABLED", "true")
    monkeypatch.setenv("BIGQUERY_AUTO_EXECUTE_MAX_COST_USD", "1.0")
    monkeypatch.setattr(
        bigquery_agent,
        "build_bigquery_sql",
        lambda _payload: {
            "answer_structured": {
                "sql": "SELECT create_date, user_cnt FROM signup_user;",
                "explanation": "ok",
            },
            "validation": {
                "success": True,
                "sql": "SELECT create_date, user_cnt FROM signup_user;",
                "total_bytes_processed": 120,
                "estimated_cost_usd": 0.0,
            },
        },
    )
    monkeypatch.setattr(
        bigquery_agent,
        "summarize_execution_result",
        lambda **_kwargs: {
            "summary": "미리보기 기준으로 특정 일자에 값이 집중됩니다.",
            "insight": "동일 일자의 복수 행이 있어 일자별 SUM 재집계가 필요해 보입니다.",
            "follow_up_questions": [
                "일자별 합계로 다시 보여줘",
                "전주 대비 증감률도 계산해줘",
                "채널별로 분해해서 보여줘",
            ],
        },
    )
    monkeypatch.setattr(
        bigquery_agent,
        "execute_bigquery_sql",
        lambda _sql: {
            "success": True,
            "job_id": "job_auto_2",
            "row_count": 4,
            "preview_rows": [
                {"create_date": "2026-02-02", "user_cnt": 5},
                {"create_date": "2026-02-02", "user_cnt": 1},
                {"create_date": "2026-02-03", "user_cnt": 3},
                {"create_date": "2026-02-03", "user_cnt": 2},
            ],
        },
    )

    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D2",
            "thread_ts": "100.4012",
            "channel_type": "im",
            "text": "전체 사용자 수 알려줘",
        }
    )

    assert result["execution"]["success"] is True
    assert ":memo: 결과 요약" in result["response_text"]
    assert ":mag: 데이터 해석" in result["response_text"]
    assert "다음 질문 제안" in result["response_text"]
    assert "일자별 합계로 다시 보여줘" in result["response_text"]


def test_run_bigquery_agent_requires_approval_when_cost_above_threshold(monkeypatch) -> None:
    monkeypatch.setenv("BIGQUERY_AUTO_EXECUTE_MAX_COST_USD", "1.0")
    monkeypatch.setattr(
        bigquery_agent,
        "build_bigquery_sql",
        lambda _payload: {
            "answer_structured": {"sql": "SELECT 2 AS value;", "explanation": "ok"},
            "validation": {
                "success": True,
                "sql": "SELECT 2 AS value;",
                "total_bytes_processed": 2048,
                "estimated_cost_usd": 1.2,
            },
        },
    )
    monkeypatch.setattr(
        bigquery_agent,
        "execute_bigquery_sql",
        lambda _sql: (_ for _ in ()).throw(AssertionError("execute should not be called")),
    )

    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D2",
            "thread_ts": "100.402",
            "channel_type": "im",
            "text": "전체 사용자 수 알려줘",
        }
    )

    assert result["action"] == "sql_generate"
    assert result["execution"]["success"] is False
    assert "자동 실행 임계값 이상" in result["execution"]["error"]
    assert result["routing"]["execution_policy"] == "approval_required"
    assert result["routing"]["execution_policy_reason"] == "cost_above_threshold"
    assert result["routing"]["estimated_cost_usd"] == 1.2


def test_run_bigquery_agent_requires_approval_when_cost_unknown(monkeypatch) -> None:
    monkeypatch.setenv("BIGQUERY_AUTO_EXECUTE_MAX_COST_USD", "1.0")
    monkeypatch.setattr(
        bigquery_agent,
        "build_bigquery_sql",
        lambda _payload: {
            "answer_structured": {"sql": "SELECT 3 AS value;", "explanation": "ok"},
            "validation": {
                "success": True,
                "sql": "SELECT 3 AS value;",
                "total_bytes_processed": 2048,
                "estimated_cost_usd": None,
            },
        },
    )
    monkeypatch.setattr(
        bigquery_agent,
        "execute_bigquery_sql",
        lambda _sql: (_ for _ in ()).throw(AssertionError("execute should not be called")),
    )

    result = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D2",
            "thread_ts": "100.403",
            "channel_type": "im",
            "text": "전체 사용자 수 알려줘",
        }
    )

    assert result["action"] == "sql_generate"
    assert result["execution"]["success"] is False
    assert "예상 비용을 계산할 수 없어" in result["execution"]["error"]
    assert result["routing"]["execution_policy"] == "approval_required"
    assert result["routing"]["execution_policy_reason"] == "estimated_cost_missing"
    assert result["routing"]["estimated_cost_usd"] is None


def test_run_bigquery_agent_executes_previous_sql_after_approval(monkeypatch) -> None:
    executed: dict[str, str] = {}

    monkeypatch.setattr(
        bigquery_agent,
        "build_bigquery_sql",
        lambda _payload: {
            "answer_structured": {"sql": "SELECT COUNT(*) FROM users", "explanation": "ok"},
            "validation": {
                "success": True,
                "sql": "SELECT COUNT(*) AS total_users FROM users;",
                "total_bytes_processed": 42,
                "estimated_cost_usd": 0.0,
            },
        },
    )
    monkeypatch.setattr(
        bigquery_agent,
        "execute_bigquery_sql",
        lambda sql: (
            executed.setdefault("sql", sql)
            and {"success": True, "job_id": "job_2", "row_count": 1, "preview_rows": [{"value": 1}]}
        ),
    )

    first = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D2",
            "thread_ts": "100.405",
            "channel_type": "im",
            "text": "전체 사용자 수 알려줘",
        }
    )
    assert first["candidate_sql"] == "SELECT COUNT(*) AS total_users FROM users;"

    second = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D2",
            "thread_ts": "100.405",
            "channel_type": "im",
            "text": "방금 만든 쿼리 실행해줘",
        }
    )

    assert second["action"] == "sql_execute"
    assert second["execution"]["success"] is False
    assert "실행 승인" in second["execution"]["error"]
    assert second["candidate_sql"] == "SELECT COUNT(*) AS total_users FROM users;"
    assert "sql" not in executed

    third = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D2",
            "thread_ts": "100.405",
            "channel_type": "im",
            "text": "실행 승인",
        }
    )

    assert third["action"] == "execution_approve"
    assert third["execution"]["success"] is True
    assert executed["sql"] == "SELECT COUNT(*) AS total_users FROM users;"


def test_run_bigquery_agent_validates_previous_sql_when_requested(monkeypatch) -> None:
    monkeypatch.setattr(
        bigquery_agent,
        "build_bigquery_sql",
        lambda _payload: {
            "answer_structured": {"sql": "SELECT COUNT(*) FROM users", "explanation": "ok"},
            "validation": {
                "success": True,
                "sql": "SELECT COUNT(*) AS total_users FROM users;",
                "total_bytes_processed": 42,
                "estimated_cost_usd": 0.0,
            },
        },
    )

    dry_run_calls: list[str] = []
    monkeypatch.setattr(
        bigquery_agent,
        "dry_run_bigquery_sql",
        lambda sql: (
            dry_run_calls.append(sql)
            or {"success": True, "sql": sql, "total_bytes_processed": 42, "estimated_cost_usd": 0.0}
        ),
    )

    first = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D2",
            "thread_ts": "100.407",
            "channel_type": "im",
            "text": "전체 사용자 수 알려줘",
        }
    )
    assert first["candidate_sql"] == "SELECT COUNT(*) AS total_users FROM users;"

    second = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D2",
            "thread_ts": "100.407",
            "channel_type": "im",
            "text": "해당 쿼리의 검증을 해주세요.",
        }
    )

    assert second["action"] == "sql_validate_explain"
    assert second["execution"] == {}
    assert second["validation"]["success"] is True
    assert second["candidate_sql"] == "SELECT COUNT(*) AS total_users FROM users;"
    assert dry_run_calls == []


def test_run_bigquery_agent_can_cancel_pending_execution(monkeypatch) -> None:
    monkeypatch.setattr(
        bigquery_agent,
        "dry_run_bigquery_sql",
        lambda _sql: {"success": True, "total_bytes_processed": 10, "estimated_cost_usd": 0.0},
    )
    monkeypatch.setattr(
        bigquery_agent,
        "execute_bigquery_sql",
        lambda _sql: (_ for _ in ()).throw(AssertionError("execute should not be called")),
    )

    first = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D2",
            "thread_ts": "100.406",
            "channel_type": "im",
            "text": "이 쿼리 실행해줘\n```sql\nSELECT 1 AS value;\n```",
        }
    )
    assert first["execution"]["success"] is False
    assert "실행 승인" in first["execution"]["error"]

    second = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D2",
            "thread_ts": "100.406",
            "channel_type": "im",
            "text": "실행 취소",
        }
    )
    assert second["action"] == "execution_cancel"
    assert second["execution"]["success"] is False
    assert "취소" in second["execution"]["error"]

    third = bigquery_agent.run_bigquery_agent(
        {
            "team_id": "T1",
            "channel_id": "D2",
            "thread_ts": "100.406",
            "channel_type": "im",
            "text": "실행 승인",
        }
    )
    assert third["execution"]["success"] is False
    assert "대기 중인 쿼리" in third["execution"]["error"]


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

    assert result["action"] == "sql_execute"
    assert result["execution"]["success"] is False
    assert "읽기 전용 SELECT" in result["execution"]["error"]
    assert execute_called is False


def test_run_bigquery_agent_blocks_execute_when_dry_run_failed(monkeypatch) -> None:
    execute_called = False

    monkeypatch.setattr(
        bigquery_agent,
        "dry_run_bigquery_sql",
        lambda _sql: {"success": False, "error": "syntax error"},
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
            "thread_ts": "100.4011",
            "channel_type": "im",
            "text": "이 쿼리 실행해줘\n```sql\nSELECT 1 AS value;\n```",
        }
    )

    assert result["action"] == "sql_execute"
    assert result["execution"]["success"] is False
    assert "dry-run 실패" in result["execution"]["error"]
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

    assert result["action"] == "sql_execute"
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
                "action": "sql_generate",
                "confidence": 0.95,
                "reason": "data task",
            }
        return {
            "action": "chat_reply",
            "confidence": 0.9,
            "reason": "chat",
        }

    monkeypatch.setattr(bigquery_agent, "plan_turn_action", fake_intent)

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
            "action": "sql_generate",
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
            "action": "sql_generate",
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
    monkeypatch.setenv("BIGQUERY_INTENT_LLM_ENABLED", "true")
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
