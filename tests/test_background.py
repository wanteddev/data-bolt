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
async def test_handle_bigquery_sql_bg_ignored_still_removes_reaction(monkeypatch) -> None:
    def fake_agent(_payload):
        return {
            "should_respond": False,
            "response_text": "",
            "generation_result": {},
            "validation": {},
            "execution": {},
        }

    calls: list[tuple[str, dict]] = []

    async def fake_run_slack_call(func, /, *args, **kwargs):
        calls.append((getattr(func, "__name__", str(func)), kwargs))
        return {}

    monkeypatch.setattr("data_bolt.tasks.bigquery_agent.run_bigquery_agent", fake_agent)
    monkeypatch.setattr(background, "_run_slack_call", fake_run_slack_call)

    result = await background.handle_bigquery_sql_bg(
        {
            "channel_id": "C1",
            "thread_ts": "111.1",
            "message_ts": "111.1",
            "text": "일반 대화",
            "include_thread_history": False,
        }
    )

    assert result["status"] == "ignored"
    assert any(kwargs.get("name") == "loading" for _, kwargs in calls)
