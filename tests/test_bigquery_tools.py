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
        tools,
        "_dry_run_callable",
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
        tools,
        "_execute_callable",
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


def test_guarded_execute_blocks_ddl_without_approval(monkeypatch) -> None:
    monkeypatch.setenv("BIGQUERY_AUTO_EXECUTE_MAX_COST_USD", "1.0")
    monkeypatch.setattr(
        tools,
        "_dry_run_callable",
        lambda _sql: {
            "success": True,
            "sql": "DROP TABLE users;",
            "estimated_cost_usd": 0.0,
            "total_bytes_processed": 1,
        },
    )
    monkeypatch.setattr(
        tools,
        "_execute_callable",
        lambda _sql: (_ for _ in ()).throw(AssertionError("execute should not be called")),
    )

    result = tools.guarded_execute_tool.run(
        action="sql_execute",
        candidate_sql="DROP TABLE users;",
        dry_run={},
        pending_execution_sql=None,
        pending_execution_dry_run=None,
    )

    assert result["can_execute"] is False
    assert result["execution"]["success"] is False
    assert "읽기 전용 SELECT" in result["execution"]["error"]
    assert result["execution_policy_reason"] == "not_read_only"


def test_guarded_execute_requires_approval_for_high_cost(monkeypatch) -> None:
    monkeypatch.setenv("BIGQUERY_AUTO_EXECUTE_MAX_COST_USD", "1.0")
    monkeypatch.setattr(
        tools,
        "_dry_run_callable",
        lambda sql: {
            "success": True,
            "sql": sql,
            "estimated_cost_usd": 1.2,
            "total_bytes_processed": 42,
        },
    )
    monkeypatch.setattr(
        tools,
        "_execute_callable",
        lambda _sql: (_ for _ in ()).throw(AssertionError("execute should not be called")),
    )

    result = tools.guarded_execute_tool.run(
        action="sql_execute",
        candidate_sql="SELECT 1;",
        dry_run={},
        pending_execution_sql=None,
        pending_execution_dry_run=None,
    )

    assert result["can_execute"] is False
    assert result["execution_policy"] == "approval_required"
    assert result["execution_policy_reason"] == "cost_above_threshold"
    assert "실행 승인" in result["execution"]["error"]


def test_guarded_execute_executes_after_approval(monkeypatch) -> None:
    monkeypatch.setenv("BIGQUERY_AUTO_EXECUTE_MAX_COST_USD", "1.0")
    monkeypatch.setattr(
        tools,
        "_execute_callable",
        lambda sql: {
            "success": True,
            "job_id": "job-approved",
            "sql": sql,
        },
    )

    result = tools.guarded_execute_tool.run(
        action="execution_approve",
        candidate_sql=None,
        dry_run={},
        pending_execution_sql="SELECT 1;",
        pending_execution_dry_run={
            "success": True,
            "sql": "SELECT 1;",
            "estimated_cost_usd": 3.0,
            "total_bytes_processed": 100,
        },
    )

    assert result["can_execute"] is True
    assert result["execution"]["success"] is True
    assert result["execution_policy"] == "auto_execute"
    assert result["execution_policy_reason"] == "user_approved"
