import pytest

from data_bolt.slack import background


@pytest.mark.asyncio
async def test_process_background_task_unknown_returns_error() -> None:
    result = await background.process_background_task({"task_type": "unknown_task", "payload": {}})

    assert result["status"] == "error"
    assert "Unknown task type" in result["message"]


@pytest.mark.asyncio
async def test_process_background_task_routes_bigquery(monkeypatch) -> None:
    called: dict[str, object] = {}

    async def fake_handler(payload):
        called["payload"] = payload
        return {"status": "ok", "result": "done"}

    monkeypatch.setattr(background, "handle_bigquery_sql_bg", fake_handler)

    payload = {"text": "hello"}
    result = await background.process_background_task(
        {"task_type": "bigquery_sql", "payload": payload}
    )

    assert result["status"] == "ok"
    assert called["payload"] == payload


@pytest.mark.asyncio
async def test_process_background_task_routes_approval(monkeypatch) -> None:
    called: dict[str, object] = {}

    async def fake_handler(payload):
        called["payload"] = payload
        return {"status": "ok", "result": "approval-done"}

    monkeypatch.setattr(background, "handle_bigquery_approval_bg", fake_handler)

    payload = {"approval_request_id": "req-1", "approved": True}
    result = await background.process_background_task(
        {"task_type": "bigquery_approval", "payload": payload}
    )

    assert result["status"] == "ok"
    assert called["payload"] == payload


@pytest.mark.asyncio
async def test_handle_bigquery_sql_bg_success_and_reaction_removed(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    async def fake_run_slack_call(func, /, *args, **kwargs):
        calls.append((getattr(func, "__name__", str(func)), kwargs))
        return {}

    async def fake_to_thread_run_sync(func, *args, **kwargs):
        if getattr(func, "__name__", "") == "run_analyst_turn":
            return {
                "should_respond": True,
                "response_text": "ok",
                "error": None,
                "generation_result": {},
                "validation": {},
                "execution": {},
            }
        return None

    monkeypatch.setattr(background, "_run_slack_call", fake_run_slack_call)
    monkeypatch.setattr(background.to_thread, "run_sync", fake_to_thread_run_sync)

    result = await background.handle_bigquery_sql_bg(
        {
            "channel_id": "C1",
            "thread_ts": "111.1",
            "message_ts": "111.1",
            "text": "지난주 가입자 수",
            "include_thread_history": False,
        }
    )

    assert result["status"] == "ok"
    assert any(kwargs.get("name") == "loading" for _, kwargs in calls)


@pytest.mark.asyncio
async def test_handle_bigquery_approval_bg_error(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    async def fake_run_slack_call(func, /, *args, **kwargs):
        calls.append((getattr(func, "__name__", str(func)), kwargs))
        return {}

    async def fake_to_thread_run_sync(func, *args, **kwargs):
        if getattr(func, "__name__", "") == "run_analyst_approval":
            return {
                "should_respond": True,
                "response_text": ":x: failed",
                "error": "approval failed",
                "generation_result": {},
                "validation": {},
                "execution": {},
            }
        return None

    monkeypatch.setattr(background, "_run_slack_call", fake_run_slack_call)
    monkeypatch.setattr(background.to_thread, "run_sync", fake_to_thread_run_sync)

    result = await background.handle_bigquery_approval_bg(
        {
            "channel_id": "C1",
            "thread_ts": "111.1",
            "approval_request_id": "req-1",
            "approved": False,
            "user_id": "U1",
        }
    )

    assert result["status"] == "error"
    assert "approval failed" in result["message"]
    assert any(name == "chat_postMessage" for name, _ in calls)


@pytest.mark.asyncio
async def test_handle_bigquery_sql_bg_skips_slack_history_when_thread_memory_exists(
    monkeypatch,
) -> None:
    fetch_calls = 0

    async def fake_fetch(*args, **kwargs):
        del args, kwargs
        nonlocal fetch_calls
        fetch_calls += 1
        return [{"role": "user", "content": "from-slack"}]

    async def fake_to_thread_run_sync(func, *args, **kwargs):
        name = getattr(func, "__name__", "")
        if name == "has_thread_memory":
            return True
        if name == "run_analyst_turn":
            return {
                "should_respond": True,
                "response_text": "ok",
                "error": None,
                "generation_result": {},
                "validation": {},
                "execution": {},
            }
        return None

    async def fake_run_slack_call(func, /, *args, **kwargs):
        del func, args, kwargs
        return {}

    monkeypatch.setenv("BIGQUERY_THREAD_MEMORY_BACKEND", "dynamodb")
    monkeypatch.setattr(background, "_fetch_thread_history", fake_fetch)
    monkeypatch.setattr(background.to_thread, "run_sync", fake_to_thread_run_sync)
    monkeypatch.setattr(background, "_run_slack_call", fake_run_slack_call)

    result = await background.handle_bigquery_sql_bg(
        {
            "channel_id": "C1",
            "thread_ts": "111.1",
            "message_ts": "111.2",
            "text": "질문",
            "team_id": "T1",
            "include_thread_history": True,
        }
    )

    assert result["status"] == "ok"
    assert fetch_calls == 0


@pytest.mark.asyncio
async def test_handle_bigquery_sql_bg_fetches_slack_history_on_thread_memory_miss(
    monkeypatch,
) -> None:
    captured_history = None

    async def fake_fetch(*args, **kwargs):
        del args, kwargs
        return [{"role": "user", "content": "from-slack"}]

    async def fake_to_thread_run_sync(func, *args, **kwargs):
        name = getattr(func, "__name__", "")
        if name == "has_thread_memory":
            return False
        if name == "run_analyst_turn":
            nonlocal captured_history
            payload = args[0]
            captured_history = payload.get("history")
            return {
                "should_respond": True,
                "response_text": "ok",
                "error": None,
                "generation_result": {},
                "validation": {},
                "execution": {},
            }
        return None

    async def fake_run_slack_call(func, /, *args, **kwargs):
        del func, args, kwargs
        return {}

    monkeypatch.setenv("BIGQUERY_THREAD_MEMORY_BACKEND", "dynamodb")
    monkeypatch.setattr(background, "_fetch_thread_history", fake_fetch)
    monkeypatch.setattr(background.to_thread, "run_sync", fake_to_thread_run_sync)
    monkeypatch.setattr(background, "_run_slack_call", fake_run_slack_call)

    result = await background.handle_bigquery_sql_bg(
        {
            "channel_id": "C1",
            "thread_ts": "111.1",
            "message_ts": "111.2",
            "text": "질문",
            "team_id": "T1",
            "include_thread_history": True,
        }
    )

    assert result["status"] == "ok"
    assert captured_history == [{"role": "user", "content": "from-slack"}]


def test_format_validation_summary_accepts_is_valid() -> None:
    text = background._format_validation_summary(
        {
            "is_valid": True,
            "total_bytes_processed": 465792,
            "estimated_cost_usd": 0.000002,
        }
    )

    assert text is not None
    assert "Dry-run passed" in text
    assert "bytes_processed=465792" in text
    assert "est_cost_usd=2e-06" in text


def test_format_bigquery_response_includes_costs_even_with_response_text() -> None:
    rendered = background._format_bigquery_response(
        {
            "response_text": "지난달 가입자 수는 18,054명입니다.",
            "validation": {
                "is_valid": True,
                "total_bytes_processed": 465792,
                "estimated_cost_usd": 0.000002,
            },
            "execution": {
                "success": True,
                "job_id": "job-1",
                "row_count": 1,
                "rows_preview": [{"new_signups": 18054}],
                "total_bytes_processed": 465792,
                "total_bytes_billed": 10485760,
                "estimated_cost_usd": 0.000002,
                "actual_cost_usd": 0.000048,
            },
        }
    )

    assert "지난달 가입자 수는 18,054명입니다." in rendered
    assert "Dry-run passed" in rendered
    assert "bytes_processed=465792" in rendered
    assert "Query executed." in rendered
    assert "bytes_billed=10485760" in rendered
    assert "actual_cost_usd=4.8e-05" in rendered
