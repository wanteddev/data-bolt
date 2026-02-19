from types import SimpleNamespace

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart

from data_bolt.tasks.analyst_agent import service
from data_bolt.tasks.analyst_agent.approval_store import ApprovalContext
from data_bolt.tasks.analyst_agent.models import AnalystReply


class FakeRunResult:
    def __init__(self, output) -> None:
        self.output = output

    def all_messages_json(self, *, output_tool_return_content=None):
        del output_tool_return_content
        return b"[]"


class FakeStore:
    def __init__(self) -> None:
        self.saved = None
        self.loaded = None
        self.deleted = None

    def save(self, context):
        self.saved = context

    def load(self, approval_request_id: str):
        del approval_request_id
        return self.loaded

    def delete(self, approval_request_id: str):
        self.deleted = approval_request_id


def test_run_analyst_turn_handles_deferred(monkeypatch) -> None:
    deferred = DeferredToolRequests(
        calls=[],
        approvals=[
            ToolCallPart(tool_name="bigquery_execute", args={"sql": "SELECT 1"}, tool_call_id="tc1")
        ],
        metadata={"tc1": {"reason": "cost", "estimated_bytes": 1000, "threshold": 100}},
    )

    fake_agent = SimpleNamespace(run_sync=lambda **kwargs: FakeRunResult(deferred))
    fake_store = FakeStore()

    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)
    monkeypatch.setattr(
        service, "DynamoApprovalStore", SimpleNamespace(from_env=lambda: fake_store)
    )

    result = service.run_analyst_turn(
        {
            "text": "이 쿼리 실행해줘",
            "user_id": "U1",
            "channel_id": "C1",
            "thread_ts": "111.1",
            "team_id": "T1",
        }
    )

    assert result["action"] == "approval_required"
    assert result["requires_approval"] is True
    assert result["approval_request_id"]
    assert fake_store.saved is not None


def test_run_analyst_approval_success(monkeypatch) -> None:
    fake_store = FakeStore()
    fake_store.loaded = ApprovalContext(
        approval_request_id="req-1",
        requester_user_id="U1",
        channel_id="C1",
        thread_ts="111.1",
        team_id="T1",
        tool_call_ids=["tc1"],
        deferred_metadata={"tc1": {"reason": "cost"}},
        session_messages_json="[]",
        created_at=1,
        expires_at=2,
    )

    fake_output = AnalystReply(message="완료", sql="SELECT 1")
    fake_agent = SimpleNamespace(run_sync=lambda **kwargs: FakeRunResult(fake_output))

    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)
    monkeypatch.setattr(
        service, "DynamoApprovalStore", SimpleNamespace(from_env=lambda: fake_store)
    )

    result = service.run_analyst_approval(
        {
            "approval_request_id": "req-1",
            "approved": True,
            "user_id": "U1",
        }
    )

    assert result["error"] is None
    assert result["response_text"] == "완료"
    assert fake_store.deleted == "req-1"


def test_run_analyst_turn_parses_string_ask_user_output(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult(
            '{"final_result_AskUser":{"message":"기간을 알려주세요"}}'
        )
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "ask_user"
    assert result["response_text"] == "기간을 알려주세요"


def test_run_analyst_turn_parses_string_ask_user_output_text_field(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult(
            '{"final_result_AskUser":{"text":"테이블명을 알려주세요"}}'
        )
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "ask_user"
    assert result["response_text"] == "테이블명을 알려주세요"


def test_run_analyst_turn_parses_string_reply_output(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult(
            '{"sql":"SELECT 1","explanation":"이 쿼리를 사용하세요"}'
        )
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "reply"
    assert result["candidate_sql"] == "SELECT 1"
    assert result["response_text"] == "이 쿼리를 사용하세요"


def test_run_analyst_turn_recovers_tool_retry_error(monkeypatch) -> None:
    def _raise_retry(**kwargs):
        del kwargs
        raise RuntimeError("Tool 'bigquery_dry_run' exceeded max retries count of 1")

    fake_agent = SimpleNamespace(run_sync=_raise_retry)
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] in {"reply", "ask_user"}
    assert result["error"] is None
