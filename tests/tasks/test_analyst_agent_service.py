from types import SimpleNamespace

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, ToolCallPart

from data_bolt.tasks.analyst_agent import service
from data_bolt.tasks.analyst_agent.approval_store import ApprovalContext
from data_bolt.tasks.analyst_agent.models import AnalystReply
from data_bolt.tasks.analyst_agent.semantic_compaction import CompactionResult
from data_bolt.tasks.analyst_agent.thread_memory_store import (
    AppendTurnResult,
    ThreadKey,
    ThreadMemoryState,
    ThreadSummary,
    ThreadTurn,
)


class FakeRunResult:
    def __init__(self, output) -> None:
        self.output = output

    def all_messages_json(self, *, output_tool_return_content=None):
        del output_tool_return_content
        return b"[]"

    def new_messages_json(self, *, output_tool_return_content=None):
        del output_tool_return_content
        return b"[]"

    def usage(self):
        return SimpleNamespace(input_tokens=10)


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


def test_run_analyst_approval_persists_followup_deferred(monkeypatch) -> None:
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
    deferred = DeferredToolRequests(
        calls=[],
        approvals=[
            ToolCallPart(tool_name="bigquery_execute", args={"sql": "SELECT 1"}, tool_call_id="tc2")
        ],
        metadata={"tc2": {"reason": "cost", "estimated_bytes": 1000, "threshold": 100}},
    )
    fake_agent = SimpleNamespace(run_sync=lambda **kwargs: FakeRunResult(deferred))

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

    assert result["action"] == "approval_required"
    assert result["approval_request_id"] != "req-1"
    assert fake_store.deleted == "req-1"
    assert fake_store.saved is not None
    assert fake_store.saved.approval_request_id == result["approval_request_id"]
    assert fake_store.saved.tool_call_ids == ["tc2"]


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


def test_run_analyst_turn_parses_string_ask_user_output_response_field(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult(
            '{"final_result_AskUser":{"response":"기간을 구체적으로 알려주세요"}}'
        )
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "ask_user"
    assert result["response_text"] == "기간을 구체적으로 알려주세요"


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


def test_run_analyst_turn_parses_string_response_field_output(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult('{"response":"일자별 유저수를 확인해볼게요."}')
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "reply"
    assert result["response_text"] == "일자별 유저수를 확인해볼게요."


def test_run_analyst_turn_parses_mapping_response_field_output(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult({"response": "관련 테이블을 먼저 찾겠습니다."})
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "reply"
    assert result["response_text"] == "관련 테이블을 먼저 찾겠습니다."


def test_run_analyst_turn_parses_mapping_reply_field_output(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult({"reply": "먼저 지표 기준을 확인해볼게요."})
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "reply"
    assert result["response_text"] == "먼저 지표 기준을 확인해볼게요."


def test_run_analyst_turn_parses_mapping_answer_field_output(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult({"answer": "해당 지표 기준으로 먼저 확인해볼게요."})
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "reply"
    assert result["response_text"] == "해당 지표 기준으로 먼저 확인해볼게요."


def test_run_analyst_turn_parses_thoughts_with_next_action_output(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult(
            {
                "thoughts": "지난달 활성 유저 흐름을 먼저 보겠습니다.",
                "next_action": {"description": "user_stats_YM에서 월별 user_active를 조회"},
            }
        )
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "reply"
    assert "지난달 활성 유저 흐름을 먼저 보겠습니다." in str(result["response_text"])
    assert "다음 단계:" in str(result["response_text"])
    assert "user_stats_YM" in str(result["response_text"])


def test_run_analyst_turn_parses_thought_with_action_sql_output(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult(
            {
                "thought": "월별 유저 통계를 먼저 확인하겠습니다.",
                "next_action": {
                    "action": "bigquery_dry_run",
                    "sql": "SELECT date, user_total FROM wanted_stats.user_stats_YM ORDER BY date DESC LIMIT 2",
                },
            }
        )
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "reply"
    assert "월별 유저 통계를 먼저 확인하겠습니다." in str(result["response_text"])
    assert "다음 단계:" in str(result["response_text"])
    assert "bigquery_dry_run" in str(result["response_text"])


def test_run_analyst_turn_parses_comment_and_tables_output(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult(
            {
                "tables": [
                    {"schema": "wanted_stats", "name": "user_stats_YM"},
                    {"schema": "wanted_stats", "name": "signup_user"},
                ],
                "comment": "유저수 분석에는 user_stats_YM이 가장 적합합니다.",
            }
        )
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "reply"
    assert result["response_text"] == "유저수 분석에는 user_stats_YM이 가장 적합합니다."


def test_run_analyst_turn_parses_error_field_output(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult({"error": "오류 상세를 먼저 알려주세요."})
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "reply"
    assert result["response_text"] == "오류 상세를 먼저 알려주세요."


def test_run_analyst_turn_parses_korean_explanation_field_output(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult(
            {"설명": "지난달 신규 가입자는 wanted_stats.signup_user.user_cnt로 확인합니다."}
        )
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "reply"
    assert "signup_user.user_cnt" in str(result["response_text"])


def test_run_analyst_turn_summarizes_korean_recommended_tables_output(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult(
            {
                "추천_테이블": [
                    {"테이블명": "wanted_stats.signup_user", "컬럼": "user_cnt"},
                    {"테이블명": "wanted_stats.user_stats_YM", "컬럼": "user_active"},
                ]
            }
        )
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "reply"
    assert "확인된 관련 테이블:" in str(result["response_text"])
    assert "wanted_stats.signup_user.user_cnt" in str(result["response_text"])


def test_run_analyst_turn_parses_wrapped_bigquery_dry_run_query_output(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult(
            {
                "bigquery_dry_run": {
                    "query": "SELECT SUM(user_cnt) FROM wanted_stats.signup_user",
                }
            }
        )
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "reply"
    assert result["candidate_sql"] == "SELECT SUM(user_cnt) FROM wanted_stats.signup_user"
    assert "요청하신 SQL을 준비했습니다." in str(result["response_text"])
    assert "```sql" in str(result["response_text"])


def test_run_analyst_turn_parses_criteria_plan_with_next_action_output(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult(
            {
                "criteria": ["신규 가입자는 signup_user 기준으로 집계합니다."],
                "plan": ["지난달 범위를 필터링해 user_cnt를 합산합니다."],
                "next_action": "지난달 신규 가입자 SQL을 생성합니다.",
            }
        )
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "reply"
    assert "signup_user" in str(result["response_text"])


def test_run_analyst_turn_maps_action_only_next_action_to_human_text(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult({"next_action": {"action": "bigquery_dry_run"}})
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "reply"
    assert "SQL 검증" in str(result["response_text"])


def test_run_analyst_turn_parses_nested_result_explanation_output(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult(
            {
                "result": {
                    "지난달_신규_가입자_수": 18054,
                    "설명": "지난달 신규 가입자 수는 18,054명입니다.",
                }
            }
        )
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "reply"
    assert result["response_text"] == "지난달 신규 가입자 수는 18,054명입니다."


def test_run_analyst_turn_recovers_tool_retry_error(monkeypatch) -> None:
    def _raise_retry(**kwargs):
        del kwargs
        raise RuntimeError("Tool 'bigquery_dry_run' exceeded max retries count of 1")

    fake_agent = SimpleNamespace(run_sync=_raise_retry)
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] in {"reply", "ask_user"}
    assert result["error"] is None


def test_run_analyst_turn_recover_includes_tool_retry_detail(monkeypatch) -> None:
    def _raise_retry(**kwargs):
        del kwargs
        raise RuntimeError(
            "Tool 'get_schema_context' exceeded max retries count of 1. "
            "Last error: missing required argument: question"
        )

    fake_agent = SimpleNamespace(run_sync=_raise_retry)
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "ask_user"
    assert "오류:" in str(result["response_text"])
    assert "question" in str(result["response_text"])


def test_run_analyst_turn_recovers_request_limit_error(monkeypatch) -> None:
    def _raise_limit(**kwargs):
        del kwargs
        raise RuntimeError("The next request would exceed the request_limit of 6")

    fake_agent = SimpleNamespace(run_sync=_raise_limit)
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn({"text": "질문", "user_id": "U1"})

    assert result["action"] == "ask_user"
    assert result["error"] is None
    assert "재시도 한도" in str(result["response_text"])


def test_run_analyst_turn_includes_runtime_trace_and_uses_history(monkeypatch) -> None:
    observed: dict[str, int] = {}

    def _fake_run_sync(**kwargs):
        observed["message_history_len"] = len(kwargs.get("message_history") or [])
        return FakeRunResult(AnalystReply(message="안내"))

    fake_agent = SimpleNamespace(run_sync=_fake_run_sync)
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    result = service.run_analyst_turn(
        {
            "text": "질문",
            "user_id": "U1",
            "history": [
                {"role": "user", "content": "안녕하세요"},
                {"role": "assistant", "content": "무엇을 도와드릴까요?"},
            ],
        }
    )

    assert observed["message_history_len"] == 2
    trace = result.get("trace")
    assert isinstance(trace, list)
    nodes = [entry.get("node") for entry in trace if isinstance(entry, dict)]
    assert "run_analyst_turn.ingest" in nodes
    assert "run_analyst_turn.history" in nodes
    assert "run_analyst_turn.apply_output" in nodes


def test_run_analyst_turn_emits_trace_callback(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult(AnalystReply(message="안내"))
    )
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)

    events: list[tuple[str, str]] = []
    result = service.run_analyst_turn(
        {"text": "질문", "user_id": "U1"},
        trace_callback=lambda node, reason: events.append((node, reason)),
    )

    assert result["action"] == "reply"
    nodes = [node for node, _ in events]
    assert "run_analyst_turn.ingest" in nodes
    assert "run_analyst_turn.agent_run_sync" in nodes
    assert "run_analyst_turn.apply_output" in nodes


def test_has_thread_memory_uses_store(monkeypatch) -> None:
    class FakeStore:
        def has_state(self, key: ThreadKey) -> bool:
            assert key.pk == "THREAD#T1#C1#111.1"
            return True

    monkeypatch.setenv("BIGQUERY_THREAD_MEMORY_BACKEND", "dynamodb")
    monkeypatch.setattr(
        service, "DynamoThreadMemoryStore", SimpleNamespace(from_env=lambda: FakeStore())
    )

    assert (
        service.has_thread_memory({"team_id": "T1", "channel_id": "C1", "thread_ts": "111.1"})
        is True
    )


def test_run_analyst_turn_appends_turn_to_thread_memory(monkeypatch) -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.append_calls = 0

        def load_state(self, key: ThreadKey) -> ThreadMemoryState:
            assert key.pk == "THREAD#T1#C1#111.1"
            return ThreadMemoryState(
                key=key,
                version=1,
                latest_turn=0,
                summary_version=0,
                summary_last_turn=0,
                prompt_tokens_ema=None,
                turns=[],
                summary=None,
            )

        def append_turn(self, **kwargs) -> AppendTurnResult:
            self.append_calls += 1
            assert kwargs["key"].pk == "THREAD#T1#C1#111.1"
            assert kwargs["usage_prompt_tokens"] == 10
            return AppendTurnResult(turn=1, version=2, prompt_tokens_ema=10.0)

    fake_store = FakeStore()
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult(AnalystReply(message="완료"))
    )

    monkeypatch.setenv("BIGQUERY_THREAD_MEMORY_BACKEND", "dynamodb")
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)
    monkeypatch.setattr(
        service, "DynamoThreadMemoryStore", SimpleNamespace(from_env=lambda: fake_store)
    )

    result = service.run_analyst_turn(
        {
            "text": "질문",
            "user_id": "U1",
            "team_id": "T1",
            "channel_id": "C1",
            "thread_ts": "111.1",
        }
    )

    assert result["action"] == "reply"
    assert fake_store.append_calls == 1
    assert result["memory_backend"] == "dynamodb"


def test_run_analyst_turn_runs_compaction_when_needed(monkeypatch) -> None:
    serialized_turn_messages = ModelMessagesTypeAdapter.dump_json(
        [ModelRequest.user_text_prompt("msg")]
    ).decode("utf-8")

    class FakeStore:
        def __init__(self) -> None:
            self.compaction_saved = 0
            self._state = ThreadMemoryState(
                key=ThreadKey(team_id="T1", channel_id="C1", thread_ts="111.1"),
                version=3,
                latest_turn=10,
                summary_version=0,
                summary_last_turn=0,
                prompt_tokens_ema=100000.0,
                turns=[
                    ThreadTurn(
                        turn=i,
                        new_messages_json=serialized_turn_messages,
                        usage_prompt_tokens=100,
                        created_at=1,
                        expires_at=9999999999,
                    )
                    for i in range(1, 11)
                ],
                summary=None,
            )

        def load_state(self, key: ThreadKey) -> ThreadMemoryState:
            assert key.pk == "THREAD#T1#C1#111.1"
            return self._state

        def save_summary_with_cas(self, **kwargs) -> bool:
            self.compaction_saved += 1
            self._state = ThreadMemoryState(
                key=self._state.key,
                version=self._state.version + 1,
                latest_turn=self._state.latest_turn,
                summary_version=self._state.summary_version + 1,
                summary_last_turn=kwargs["source_turn_end"],
                prompt_tokens_ema=self._state.prompt_tokens_ema,
                turns=self._state.turns,
                summary=ThreadSummary(
                    version=self._state.summary_version + 1,
                    summary_text=kwargs["summary_text"],
                    summary_struct_json=kwargs["summary_struct_json"],
                    source_turn_start=kwargs["source_turn_start"],
                    source_turn_end=kwargs["source_turn_end"],
                    created_at=1,
                    expires_at=9999999999,
                ),
            )
            return True

        def append_turn(self, **kwargs) -> AppendTurnResult:
            del kwargs
            return AppendTurnResult(turn=11, version=5, prompt_tokens_ema=100000.0)

    fake_store = FakeStore()
    fake_agent = SimpleNamespace(
        run_sync=lambda **kwargs: FakeRunResult(AnalystReply(message="완료"))
    )

    monkeypatch.setenv("BIGQUERY_THREAD_MEMORY_BACKEND", "dynamodb")
    monkeypatch.setattr(service, "_get_agent", lambda: fake_agent)
    monkeypatch.setattr(
        service, "DynamoThreadMemoryStore", SimpleNamespace(from_env=lambda: fake_store)
    )
    monkeypatch.setattr(
        service,
        "compact_history",
        lambda **kwargs: CompactionResult(
            summary_text="요약",
            summary_struct={
                "durable_facts": ["사용자 이름: 우징"],
                "open_tasks": [],
                "user_preferences": [],
                "sql_artifacts": [],
                "constraints": [],
                "risk_flags": [],
                "summary_text": "요약",
            },
        ),
    )
    monkeypatch.setattr(service, "estimate_tokens", lambda **kwargs: 100000)
    monkeypatch.setattr(service, "should_compact", lambda **kwargs: True)

    result = service.run_analyst_turn(
        {
            "text": "질문",
            "user_id": "U1",
            "team_id": "T1",
            "channel_id": "C1",
            "thread_ts": "111.1",
        }
    )

    assert result["action"] == "reply"
    assert fake_store.compaction_saved == 1
    trace = result.get("trace") or []
    reasons = [entry.get("reason") for entry in trace if isinstance(entry, dict)]
    assert any("semantic compaction 적용 완료" in str(reason) for reason in reasons)
