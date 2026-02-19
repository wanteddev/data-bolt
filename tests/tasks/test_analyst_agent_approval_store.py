from data_bolt.tasks.analyst_agent.approval_store import (
    ApprovalContext,
    DynamoApprovalStore,
    build_approval_context_timestamps,
)


class FakeTable:
    def __init__(self) -> None:
        self.items: dict[str, dict] = {}

    def put_item(self, Item):
        self.items[Item["approval_request_id"]] = Item

    def get_item(self, Key):
        item = self.items.get(Key["approval_request_id"])
        return {"Item": item} if item else {}

    def delete_item(self, Key):
        self.items.pop(Key["approval_request_id"], None)


class FakeDynamoResource:
    def __init__(self, table: FakeTable) -> None:
        self._table = table

    def Table(self, _table_name: str):
        return self._table


def test_approval_store_save_load_delete(monkeypatch) -> None:
    table = FakeTable()
    monkeypatch.setattr(
        "data_bolt.tasks.analyst_agent.approval_store.boto3.resource",
        lambda _service: FakeDynamoResource(table),
    )

    store = DynamoApprovalStore("approval-table")
    context = ApprovalContext(
        approval_request_id="req-1",
        requester_user_id="U1",
        channel_id="C1",
        thread_ts="111.1",
        team_id="T1",
        tool_call_ids=["tool-1"],
        deferred_metadata={"tool-1": {"reason": "cost"}},
        session_messages_json="[]",
        created_at=100,
        expires_at=200,
    )

    store.save(context)
    loaded = store.load("req-1")

    assert loaded is not None
    assert loaded.requester_user_id == "U1"

    store.delete("req-1")
    assert store.load("req-1") is None


def test_build_approval_context_timestamps_default(monkeypatch) -> None:
    monkeypatch.delenv("BIGQUERY_APPROVAL_TTL_SECONDS", raising=False)

    created_at, expires_at = build_approval_context_timestamps()

    assert expires_at > created_at
