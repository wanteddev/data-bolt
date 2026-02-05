from litestar.testing import TestClient
from slack_bolt import BoltResponse
from slack_bolt.async_app import AsyncBoltRequest

from data_bolt.app import app


def test_health_endpoints() -> None:
    with TestClient(app=app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json() == {"ok": "web"}

        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy"}


def test_slack_events_dispatch_is_async(monkeypatch) -> None:
    seen: dict[str, object | None] = {"request": None}

    async def fake_dispatch(request) -> BoltResponse:
        seen["request"] = request
        return BoltResponse(status=200, body="ok", headers={"Content-Type": "text/plain"})

    monkeypatch.setattr("data_bolt.app.slack_app.async_dispatch", fake_dispatch)

    with TestClient(app=app) as client:
        resp = client.post(
            "/slack/events",
            content=b'{"event_id":"evt-1"}',
            headers={"Content-Type": "application/json"},
        )

    assert resp.status_code == 200
    assert resp.text == "ok"
    assert isinstance(seen["request"], AsyncBoltRequest)
    assert seen["request"].raw_body == '{"event_id":"evt-1"}'
