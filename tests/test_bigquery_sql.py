from typing import Any

from data_bolt.tasks import bigquery_sql


def test_build_bigquery_sql_returns_structured_response(monkeypatch) -> None:
    def fake_generate(*_args, **_kwargs) -> dict:
        content = (
            "{"
            '"sql": "SELECT 1 AS one;",'
            '"explanation": "simple",'
            '"assumptions": "",'
            '"validation_steps": []'
            "}"
        )
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(bigquery_sql, "generate_bigquery_response", fake_generate)
    monkeypatch.setattr(
        bigquery_sql,
        "_dry_run_sql",
        lambda _sql: (True, {"total_bytes_processed": 1024, "job_id": "job1", "cache_hit": False}),
    )

    result = bigquery_sql.build_bigquery_sql(
        {"text": "count users", "table_info": "ddl", "glossary_info": "glossary"}
    )

    assert result["answer_structured"]["sql"] == "SELECT 1 AS one;"
    assert result["answer_structured"]["explanation"] == "simple"
    assert result["validation"]["success"] is True


def test_build_bigquery_sql_parses_laas_content_blocks(monkeypatch) -> None:
    def fake_generate(*_args, **_kwargs) -> dict:
        content = (
            "{"
            '"sql": "SELECT COUNT(*) AS cnt FROM users;",'
            '"explanation": "laas blocks",'
            '"assumptions": "",'
            '"validation_steps": []'
            "}"
        )
        return {"choices": [{"message": {"content": [{"type": "text", "text": content}]}}]}

    monkeypatch.setattr(bigquery_sql, "generate_bigquery_response", fake_generate)
    monkeypatch.setattr(
        bigquery_sql,
        "_dry_run_sql",
        lambda _sql: (True, {"total_bytes_processed": 1024, "job_id": "job1", "cache_hit": False}),
    )

    result = bigquery_sql.build_bigquery_sql(
        {"text": "count users", "table_info": "ddl", "glossary_info": "glossary"}
    )

    assert result["answer_structured"]["sql"] == "SELECT COUNT(*) AS cnt FROM users;"
    assert result["answer_structured"]["explanation"] == "laas blocks"
    assert result["validation"]["success"] is True


def test_adapt_laas_response_for_agent_supports_output_blocks() -> None:
    raw_response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "{"
                            '"sql":"SELECT 1 AS one;",'
                            '"explanation":"from output block",'
                            '"assumptions":"",'
                            '"validation_steps":[]'
                            "}"
                        ),
                    }
                ],
            }
        ]
    }

    adapted = bigquery_sql.adapt_laas_response_for_agent(
        raw_response,
        "bigquery_sql_generation",
    )

    assert adapted["answer_structured"]["sql"] == "SELECT 1 AS one;"
    assert adapted["answer_structured"]["explanation"] == "from output block"


def test_adapt_laas_response_for_agent_keeps_empty_sql_without_fallback() -> None:
    raw_response = {
        "choices": [
            {
                "message": {
                    "content": (
                        "{"
                        '"sql":"",'
                        '"explanation":"인사말 응답",'
                        '"assumptions":[],'
                        '"validation_steps":[]'
                        "}"
                    )
                }
            }
        ]
    }

    adapted = bigquery_sql.adapt_laas_response_for_agent(
        raw_response,
        "general_chat",
    )

    assert adapted["answer_structured"]["sql"] is None
    assert adapted["answer_structured"]["explanation"] == "인사말 응답"
    assert adapted["choices"][0]["message"]["content"] == ""


def test_build_bigquery_sql_returns_error_on_failure(monkeypatch) -> None:
    def fake_generate(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(bigquery_sql, "generate_bigquery_response", fake_generate)

    result = bigquery_sql.build_bigquery_sql({"text": "count users"})

    assert "error" in result


def test_build_bigquery_sql_refines_when_initial_dry_run_fails(monkeypatch) -> None:
    def fake_generate(*_args, **_kwargs) -> dict:
        content = (
            '{"sql": "SELEC 1;","explanation": "bad sql","assumptions": "","validation_steps": []}'
        )
        return {"choices": [{"message": {"content": content}}]}

    attempts: dict[str, int] = {"count": 0}

    def fake_dry_run(sql: str):
        attempts["count"] += 1
        if "SELEC 1" in sql:
            return False, {"error": "Syntax error", "total_bytes_processed": None}
        return True, {"total_bytes_processed": 1024, "job_id": "job2", "cache_hit": False}

    monkeypatch.setattr(bigquery_sql, "generate_bigquery_response", fake_generate)
    monkeypatch.setattr(bigquery_sql, "_dry_run_sql", fake_dry_run)
    monkeypatch.setattr(bigquery_sql, "refine_bigquery_sql", lambda **_kwargs: "SELECT 1;")
    monkeypatch.setenv("BIGQUERY_REFINE_MAX_ATTEMPTS", "1")

    result = bigquery_sql.build_bigquery_sql({"text": "count users", "table_info": "ddl"})

    assert attempts["count"] >= 2
    assert result["validation"]["success"] is True
    assert result["validation"]["refined"] is True


def test_build_bigquery_sql_skips_rag_for_general_chat(monkeypatch) -> None:
    called = {"rag": False}

    def fake_rag(_question: str) -> dict:
        called["rag"] = True
        return {"table_info": "x", "glossary_info": "y", "meta": {"attempted": True}}

    def fake_generate(*_args, **_kwargs) -> dict:
        content = '{"sql": "","explanation": "안녕하세요","assumptions": [],"validation_steps": []}'
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(bigquery_sql, "_collect_rag_context", fake_rag)
    monkeypatch.setattr(bigquery_sql, "generate_bigquery_response", fake_generate)
    result = bigquery_sql.build_bigquery_sql({"text": "안녕", "instruction_type": "general_chat"})

    assert called["rag"] is False
    assert result["meta"]["context_source"] == "provided"


def test_llm_chat_completion_defaults_to_laas(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_laas_post(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        captured["path"] = path
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {"choices": []}

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(bigquery_sql, "_laas_post", fake_laas_post)
    monkeypatch.setattr(
        bigquery_sql, "_openai_compatible_post", lambda *_args, **_kwargs: {"unexpected": True}
    )

    response = bigquery_sql._llm_chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        timeout=12.5,
    )

    assert response == {"choices": []}
    assert captured["path"] == "/api/preset/v2/chat/completions"
    assert captured["payload"]["hash"] == bigquery_sql.LAAS_EMPTY_PRESET_HASH
    assert captured["payload"]["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["timeout"] == 12.5


def test_llm_chat_completion_openai_compatible_uses_default_model(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_openai_post(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        captured["path"] = path
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_OPENAI_BASE_URL", "https://example.test")
    monkeypatch.setenv("LLM_OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("LLM_OPENAI_MODEL", raising=False)
    monkeypatch.setattr(bigquery_sql, "_openai_compatible_post", fake_openai_post)
    monkeypatch.setattr(bigquery_sql, "_laas_post", lambda *_args, **_kwargs: {"unexpected": True})

    response = bigquery_sql._llm_chat_completion(
        messages=[{"role": "system", "content": "x"}],
        timeout=3.0,
    )

    assert response == {"choices": [{"message": {"content": "ok"}}]}
    assert captured["path"] == "/chat/completions"
    assert captured["payload"]["model"] == "glm-4.7"
    assert captured["payload"]["messages"] == [{"role": "system", "content": "x"}]
    assert captured["timeout"] == 3.0


def test_llm_chat_completion_anthropic_compatible_formats_system(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_anthropic_post(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        captured["path"] = path
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setenv("LLM_PROVIDER", "anthropic_compatible")
    monkeypatch.setenv("LLM_ANTHROPIC_BASE_URL", "https://example.test")
    monkeypatch.setenv("LLM_ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("LLM_ANTHROPIC_MODEL", raising=False)
    monkeypatch.setattr(bigquery_sql, "_anthropic_compatible_post", fake_anthropic_post)

    response = bigquery_sql._llm_chat_completion(
        messages=[
            {"role": "system", "content": "sys-a"},
            {"role": "system", "content": "sys-b"},
            {"role": "user", "content": "hello"},
        ],
        timeout=7.0,
    )

    assert response == {"content": [{"type": "text", "text": "ok"}]}
    assert captured["path"] == "/v1/messages"
    assert captured["payload"]["model"] == "claude-haiku-4-5-20251001"
    assert captured["payload"]["system"] == "sys-a\n\nsys-b"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["timeout"] == 7.0


def test_classify_intent_with_laas_parses_anthropic_content(monkeypatch) -> None:
    monkeypatch.setattr(
        bigquery_sql,
        "_llm_chat_completion",
        lambda **_kwargs: {
            "content": [
                {
                    "type": "text",
                    "text": (
                        '{"intent":"free_chat","confidence":0.9,'
                        '"reason":"anthropic","actions":["none"]}'
                    ),
                }
            ]
        },
    )

    result = bigquery_sql.classify_intent_with_laas(
        text="안녕",
        history=[],
        channel_type="im",
        is_mention=False,
        is_thread_followup=False,
    )

    assert result["intent"] == "free_chat"
    assert result["actions"] == ["none"]


def test_build_bigquery_sql_meta_uses_llm(monkeypatch) -> None:
    def fake_generate(*_args, **_kwargs) -> dict:
        return {
            "model": "glm-4.7",
            "choices": [{"message": {"content": '{"sql":"SELECT 1;","explanation":"ok"}'}}],
        }

    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setattr(bigquery_sql, "generate_bigquery_response", fake_generate)
    monkeypatch.setattr(
        bigquery_sql,
        "_dry_run_sql",
        lambda _sql: (True, {"total_bytes_processed": 0, "job_id": "job", "cache_hit": False}),
    )

    result = bigquery_sql.build_bigquery_sql({"text": "count users", "table_info": "ddl"})

    assert result["meta"]["llm"]["provider"] == "openai_compatible"
    assert result["meta"]["llm"]["model"] == "glm-4.7"
    assert result["meta"]["laas"]["called"] is False


def test_get_llm_timeout_seconds_uses_defaults(monkeypatch) -> None:
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LLM_TIMEOUT_INTENT_SECONDS", raising=False)
    monkeypatch.delenv("LLM_TIMEOUT_CHAT_PLANNER_SECONDS", raising=False)
    monkeypatch.delenv("LLM_TIMEOUT_GENERATION_SECONDS", raising=False)
    monkeypatch.delenv("LLM_TIMEOUT_REFINE_SECONDS", raising=False)

    assert bigquery_sql._get_llm_timeout_seconds("intent") == 45.0
    assert bigquery_sql._get_llm_timeout_seconds("planner") == 45.0
    assert bigquery_sql._get_llm_timeout_seconds("generation") == 60.0
    assert bigquery_sql._get_llm_timeout_seconds("refine") == 60.0


def test_get_llm_timeout_seconds_prefers_specific_over_common(monkeypatch) -> None:
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("LLM_TIMEOUT_INTENT_SECONDS", "30")
    monkeypatch.setenv("LLM_TIMEOUT_CHAT_PLANNER_SECONDS", "35")
    monkeypatch.setenv("LLM_TIMEOUT_GENERATION_SECONDS", "120")
    monkeypatch.setenv("LLM_TIMEOUT_REFINE_SECONDS", "150")

    assert bigquery_sql._get_llm_timeout_seconds("intent") == 30.0
    assert bigquery_sql._get_llm_timeout_seconds("planner") == 35.0
    assert bigquery_sql._get_llm_timeout_seconds("generation") == 120.0
    assert bigquery_sql._get_llm_timeout_seconds("refine") == 150.0


def test_parse_json_response_supports_fenced_json() -> None:
    content = (
        '```json\n{"sql":"","explanation":"안녕하세요","assumptions":[],"validation_steps":[]}\n```'
    )
    parsed = bigquery_sql._parse_json_response(content)
    assert parsed["sql"] == ""
    assert parsed["explanation"] == "안녕하세요"
