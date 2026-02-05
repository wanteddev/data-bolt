from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from data_bolt.slack import handlers


@pytest.mark.asyncio
async def test_handle_message_dm_says_result() -> None:
    ack = AsyncMock()
    say = AsyncMock()
    event = {"channel_type": "im", "user": "U1", "text": "hi", "ts": "123"}

    await handlers.handle_message(event=event, say=say, ack=ack)

    ack.assert_awaited_once()
    say.assert_awaited_once()
    assert "I processed your message" in say.call_args.args[0]


@pytest.mark.asyncio
async def test_handle_message_non_im_ignored() -> None:
    ack = AsyncMock()
    say = AsyncMock()
    event = {"channel_type": "channel", "user": "U1", "text": "hi", "ts": "123"}

    await handlers.handle_message(event=event, say=say, ack=ack)

    ack.assert_awaited_once()
    say.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_message_bot_ignored() -> None:
    ack = AsyncMock()
    say = AsyncMock()
    event = {"channel_type": "im", "bot_id": "B1", "text": "hi", "ts": "123"}

    await handlers.handle_message(event=event, say=say, ack=ack)

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
