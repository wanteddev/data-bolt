from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from data_bolt.slack import handlers


@pytest.mark.asyncio
async def test_handle_message_dm_triggers_background(monkeypatch) -> None:
    ack = AsyncMock()
    say = AsyncMock()
    context = SimpleNamespace(client=SimpleNamespace(reactions_add=AsyncMock()))
    event = {"channel_type": "im", "user": "U1", "text": "hi", "ts": "123"}

    called: dict[str, object] = {}

    async def fake_run_sync(func, *args, **kwargs):
        called["func"] = func
        called["args"] = args
        called["kwargs"] = kwargs
        return None

    monkeypatch.setattr(handlers.to_thread, "run_sync", fake_run_sync)

    await handlers.handle_message(event=event, say=say, context=context, ack=ack)

    ack.assert_awaited_once()
    say.assert_not_awaited()
    assert called["func"] is handlers.invoke_background
    assert called["args"][0] == "bigquery_sql"


@pytest.mark.asyncio
async def test_handle_message_non_im_ignored() -> None:
    ack = AsyncMock()
    say = AsyncMock()
    context = SimpleNamespace(client=SimpleNamespace(reactions_add=AsyncMock()))
    event = {"channel_type": "channel", "user": "U1", "text": "hi", "ts": "123"}

    await handlers.handle_message(event=event, say=say, context=context, ack=ack)

    ack.assert_awaited_once()
    say.assert_not_awaited()
    context.client.reactions_add.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_message_bot_ignored() -> None:
    ack = AsyncMock()
    say = AsyncMock()
    context = SimpleNamespace(client=SimpleNamespace(reactions_add=AsyncMock()))
    event = {"channel_type": "im", "bot_id": "B1", "text": "hi", "ts": "123"}

    await handlers.handle_message(event=event, say=say, context=context, ack=ack)

    ack.assert_awaited_once()
    say.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_app_mention_triggers_background(monkeypatch) -> None:
    ack = AsyncMock()
    say = AsyncMock()
    reactions_add = AsyncMock()
    context = SimpleNamespace(client=SimpleNamespace(reactions_add=reactions_add))

    called: dict[str, object] = {}

    async def fake_run_sync(func, *args, **kwargs):
        called["func"] = func
        called["args"] = args
        called["kwargs"] = kwargs
        return None

    monkeypatch.setattr(handlers.to_thread, "run_sync", fake_run_sync)

    event = {"user": "U1", "text": "sql pls", "channel": "C1", "ts": "123"}

    await handlers.handle_app_mention(
        event=event,
        say=say,
        context=context,
        ack=ack,
    )

    ack.assert_awaited_once()
    reactions_add.assert_awaited_once()
    say.assert_not_awaited()

    assert called["func"] is handlers.invoke_background
    assert called["args"][0] == "bigquery_sql"
    payload = called["args"][1]
    assert payload["channel_id"] == "C1"
    assert payload["message_ts"] == "123"
    assert payload["text"] == "sql pls"
    assert payload["is_mention"] is True


@pytest.mark.asyncio
async def test_handle_approve_action_invokes_background(monkeypatch) -> None:
    ack = AsyncMock()

    called: dict[str, object] = {}

    async def fake_run_sync(func, *args, **kwargs):
        called["func"] = func
        called["args"] = args
        return None

    monkeypatch.setattr(handlers.to_thread, "run_sync", fake_run_sync)

    body = {
        "actions": [{"value": "req-1"}],
        "team": {"id": "T1"},
        "user": {"id": "U1"},
        "channel": {"id": "C1"},
        "container": {"message_ts": "100.1", "channel_id": "C1"},
        "message": {"thread_ts": "100.1", "ts": "100.1"},
    }

    await handlers.handle_approve_execute_action(ack=ack, body=body)

    ack.assert_awaited_once()
    assert called["func"] is handlers.invoke_background
    assert called["args"][0] == "bigquery_approval"
    payload = called["args"][1]
    assert payload["approval_request_id"] == "req-1"
    assert payload["approved"] is True


@pytest.mark.asyncio
async def test_handle_deny_action_invokes_background(monkeypatch) -> None:
    ack = AsyncMock()

    called: dict[str, object] = {}

    async def fake_run_sync(func, *args, **kwargs):
        called["func"] = func
        called["args"] = args
        return None

    monkeypatch.setattr(handlers.to_thread, "run_sync", fake_run_sync)

    body = {
        "actions": [{"value": "req-2"}],
        "team": {"id": "T1"},
        "user": {"id": "U1"},
        "channel": {"id": "C1"},
        "container": {"message_ts": "100.1", "channel_id": "C1"},
        "message": {"thread_ts": "100.1", "ts": "100.1"},
    }

    await handlers.handle_deny_execute_action(ack=ack, body=body)

    ack.assert_awaited_once()
    assert called["func"] is handlers.invoke_background
    assert called["args"][0] == "bigquery_approval"
    payload = called["args"][1]
    assert payload["approval_request_id"] == "req-2"
    assert payload["approved"] is False
