from decimal import Decimal

import boto3
from botocore.exceptions import ClientError

from data_bolt.tasks.analyst_agent.thread_memory_store import (
    DynamoThreadMemoryStore,
    ThreadKey,
)


class _FakeTable:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self.update_calls = 0
        self.put_calls = 0
        self.fail_first_put = False
        self.fail_update_conditional = False

    def query(self, **kwargs):
        del kwargs
        return {"Items": list(self.items)}

    def update_item(self, **kwargs):
        self.update_calls += 1
        if self.fail_update_conditional:
            raise ClientError(
                {"Error": {"Code": "ConditionalCheckFailedException", "Message": "mismatch"}},
                "UpdateItem",
            )
        return {
            "Attributes": {
                "version": self.update_calls,
                "latest_turn": self.update_calls,
                "summary_version": 0,
                "summary_last_turn": 0,
                "prompt_tokens_ema": Decimal("10.0"),
            }
        }

    def put_item(self, **kwargs):
        self.put_calls += 1
        if self.fail_first_put and self.put_calls == 1:
            raise ClientError(
                {"Error": {"Code": "ConditionalCheckFailedException", "Message": "duplicate"}},
                "PutItem",
            )
        item = kwargs["Item"]
        self.items.append(item)
        return {}

    def delete_item(self, **kwargs):
        key = kwargs["Key"]
        self.items = [
            item
            for item in self.items
            if not (item.get("PK") == key.get("PK") and item.get("SK") == key.get("SK"))
        ]
        return {}


class _FakeResource:
    def __init__(self, table: _FakeTable) -> None:
        self._table = table

    def Table(self, table_name: str):
        assert table_name == "thread-memory"
        return self._table


def test_load_state_parses_meta_turn_summary(monkeypatch) -> None:
    table = _FakeTable()
    table.items = [
        {
            "PK": "THREAD#T1#C1#111.1",
            "SK": "META",
            "version": 3,
            "latest_turn": 2,
            "summary_version": 1,
            "summary_last_turn": 1,
            "prompt_tokens_ema": Decimal("123.0"),
            "expires_at": 9999999999,
        },
        {
            "PK": "THREAD#T1#C1#111.1",
            "SK": "SUMMARY#00000001",
            "summary_text": "요약",
            "summary_struct_json": '{"summary_text":"요약"}',
            "source_turn_start": 1,
            "source_turn_end": 1,
            "created_at": 1,
            "expires_at": 9999999999,
        },
        {
            "PK": "THREAD#T1#C1#111.1",
            "SK": "TURN#00000002",
            "new_messages_json": "[]",
            "usage_prompt_tokens": 42,
            "created_at": 1,
            "expires_at": 9999999999,
        },
    ]
    monkeypatch.setattr(boto3, "resource", lambda *_args, **_kwargs: _FakeResource(table))

    store = DynamoThreadMemoryStore("thread-memory")
    state = store.load_state(ThreadKey(team_id="T1", channel_id="C1", thread_ts="111.1"))

    assert state is not None
    assert state.version == 3
    assert state.latest_turn == 2
    assert state.summary is not None
    assert state.summary.summary_text == "요약"
    assert state.prompt_tokens_ema == 123.0


def test_append_turn_retries_when_conditional_check_fails(monkeypatch) -> None:
    table = _FakeTable()
    table.fail_first_put = True
    monkeypatch.setattr(boto3, "resource", lambda *_args, **_kwargs: _FakeResource(table))

    store = DynamoThreadMemoryStore("thread-memory")
    result = store.append_turn(
        key=ThreadKey(team_id="T1", channel_id="C1", thread_ts="111.1"),
        new_messages_json="[]",
        prompt_tokens_ema=10.0,
        usage_prompt_tokens=10,
    )

    assert result.turn == 2
    assert table.update_calls == 2


def test_save_summary_with_cas_returns_false_on_conditional_failure(monkeypatch) -> None:
    table = _FakeTable()
    table.fail_update_conditional = True
    monkeypatch.setattr(boto3, "resource", lambda *_args, **_kwargs: _FakeResource(table))

    store = DynamoThreadMemoryStore("thread-memory")
    ok = store.save_summary_with_cas(
        key=ThreadKey(team_id="T1", channel_id="C1", thread_ts="111.1"),
        expected_version=1,
        expected_summary_version=0,
        summary_text="요약",
        summary_struct_json='{"summary_text":"요약"}',
        source_turn_start=2,
        source_turn_end=5,
        prompt_tokens_ema=100.0,
    )

    assert ok is False
