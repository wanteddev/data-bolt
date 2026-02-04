import asyncio
from unittest.mock import Mock

import pytest

from data_bolt.slack import background


@pytest.mark.asyncio
async def test_handle_dm_message_bg_uses_threaded_slack_call(monkeypatch) -> None:
    async def fast_sleep(*_args, **_kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    mock_chat = Mock(return_value={"ok": True})
    monkeypatch.setattr(background.slack_client, "chat_postMessage", mock_chat)

    called = {"count": 0}
    orig_run_sync = background.to_thread.run_sync

    async def wrapped(func, *args, **kwargs):
        called["count"] += 1
        return await orig_run_sync(func, *args, **kwargs)

    monkeypatch.setattr(background.to_thread, "run_sync", wrapped)

    payload = {"user_id": "U1", "channel_id": "C1", "text": "hi", "ts": "123"}
    result = await background._handle_dm_message_bg(payload)

    assert result["status"] == "ok"
    assert called["count"] >= 1
    assert mock_chat.call_count == 1
