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
