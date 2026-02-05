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
