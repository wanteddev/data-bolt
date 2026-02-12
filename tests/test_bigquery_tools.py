from data_bolt.tasks.bigquery import tools


def test_rag_context_tool_delegates_to_rag(monkeypatch) -> None:
    monkeypatch.setattr(
        tools.rag,
        "_collect_rag_context",
        lambda question: {
            "table_info": f"ddl:{question}",
            "glossary_info": "glossary",
            "meta": {"attempted": True},
        },
    )

    result = tools.RagContextTool().run(question="가입자 수")

    assert result["table_info"] == "ddl:가입자 수"
    assert result["meta"]["attempted"] is True


def test_dry_run_tool_delegates_to_execution(monkeypatch) -> None:
    monkeypatch.setattr(
        tools.execution,
        "dry_run_bigquery_sql",
        lambda sql: {
            "success": True,
            "sql": sql,
            "estimated_cost_usd": 0.12,
            "total_bytes_processed": 10,
        },
    )

    result = tools.DryRunTool().run(sql="SELECT 1;")

    assert result["success"] is True
    assert result["sql"] == "SELECT 1;"
    assert result["estimated_cost_usd"] == 0.12


def test_execute_query_tool_delegates_to_execution(monkeypatch) -> None:
    monkeypatch.setattr(
        tools.execution,
        "execute_bigquery_sql",
        lambda sql: {
            "success": True,
            "job_id": "job-1",
            "row_count": 1,
            "preview_rows": [{"v": 1}],
        },
    )

    result = tools.ExecuteQueryTool().run(sql="SELECT 1;")

    assert result["success"] is True
    assert result["job_id"] == "job-1"
    assert result["preview_rows"] == [{"v": 1}]
