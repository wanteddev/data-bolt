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
    monkeypatch.delenv("BIGQUERY_DRYRUN_URL", raising=False)

    result = bigquery_sql.build_bigquery_sql(
        {"text": "count users", "table_info": "ddl", "glossary_info": "glossary"}
    )

    assert result["answer_structured"]["sql"] == "SELECT 1 AS one;"
    assert result["answer_structured"]["explanation"] == "simple"
    assert "validation" not in result


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
    monkeypatch.delenv("BIGQUERY_DRYRUN_URL", raising=False)

    result = bigquery_sql.build_bigquery_sql(
        {"text": "count users", "table_info": "ddl", "glossary_info": "glossary"}
    )

    assert result["answer_structured"]["sql"] == "SELECT COUNT(*) AS cnt FROM users;"
    assert result["answer_structured"]["explanation"] == "laas blocks"


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


def test_build_bigquery_sql_returns_error_on_failure(monkeypatch) -> None:
    def fake_generate(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(bigquery_sql, "generate_bigquery_response", fake_generate)

    result = bigquery_sql.build_bigquery_sql({"text": "count users"})

    assert "error" in result
