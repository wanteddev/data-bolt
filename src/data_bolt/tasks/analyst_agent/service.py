"""Public service entrypoints for the PydanticAI-based analyst agent."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic_ai import Agent

from .approval_store import ApprovalContext, DynamoApprovalStore
from .deps import AnalystDeps
from .model_factory import build_model_for_env
from .runtime import approval_flow, config, recovery, result_contract, thread_context
from .semantic_compaction import (
    compact_history as _compact_history,
)
from .semantic_compaction import (
    estimate_tokens as _estimate_tokens,
)
from .semantic_compaction import (
    load_compaction_config,
)
from .semantic_compaction import (
    should_compact as _should_compact,
)
from .thread_memory_store import DynamoThreadMemoryStore
from .tools import register_analyst_tools

_agent: Agent[AnalystDeps, Any] | None = None
type TraceCallback = Callable[[str, str], None]

# Keep these aliases patchable from tests while moving the implementation to runtime modules.
compact_history = _compact_history
estimate_tokens = _estimate_tokens
should_compact = _should_compact


def _build_agent() -> Agent[AnalystDeps, Any]:
    agent = Agent(
        build_model_for_env(),
        deps_type=AnalystDeps,
        output_type=config.OUTPUT_TYPES,
        instructions=config.ANALYST_SYSTEM_INSTRUCTIONS,
        output_retries=2,
        defer_model_check=True,
    )
    register_analyst_tools(agent)
    return agent


def _get_agent() -> Agent[AnalystDeps, Any]:
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


def _build_deps(payload: Mapping[str, Any]) -> AnalystDeps:
    return config.build_deps(payload)


def _base_result() -> dict[str, Any]:
    return result_contract.base_result()


def _append_trace(trace: list[dict[str, str]], node: str, reason: str) -> None:
    trace.append({"node": node, "reason": reason})


def _with_trace(result: dict[str, Any], trace: list[dict[str, str]]) -> dict[str, Any]:
    if trace:
        result["trace"] = trace
    return result


def _save_approval_context(context: ApprovalContext) -> None:
    DynamoApprovalStore.from_env().save(context)


def has_thread_memory(payload: Mapping[str, Any]) -> bool:
    """Return whether the thread has persisted memory state."""
    return thread_context.has_thread_memory(
        payload,
        thread_store_factory=lambda: DynamoThreadMemoryStore.from_env(),
    )


def run_analyst_turn(
    payload: dict[str, Any],
    trace_callback: TraceCallback | None = None,
) -> dict[str, Any]:
    """Run one analyst turn and return Slack-friendly result contract."""
    trace: list[dict[str, str]] = []

    def emit_trace(node: str, reason: str) -> None:
        _append_trace(trace, node, reason)
        if trace_callback is None:
            return
        try:
            trace_callback(node, reason)
        except Exception:
            return

    text = str(payload.get("text") or payload.get("question") or "").strip()
    history_raw = payload.get("history")
    history_count = len(history_raw) if isinstance(history_raw, list) else 0
    emit_trace(
        "run_analyst_turn.ingest",
        f"입력을 정규화했습니다. text_len={len(text)}, history_items={history_count}",
    )

    if not text:
        result = {
            **_base_result(),
            "should_respond": False,
            "response_text": "",
            "action": "ignore",
        }
        emit_trace("run_analyst_turn.finish", "빈 입력으로 ignore를 반환합니다.")
        return _with_trace(result, trace)

    compaction_config = load_compaction_config()
    prepared_context = thread_context.prepare_thread_context(
        payload=payload,
        history_raw=history_raw,
        compaction_config=compaction_config,
        thread_store_factory=lambda: DynamoThreadMemoryStore.from_env(),
        emit_trace=emit_trace,
    )

    prepared_context, _ = thread_context.maybe_compact_thread_context(
        context=prepared_context,
        text=text,
        compaction_config=compaction_config,
        estimate_tokens_fn=estimate_tokens,
        should_compact_fn=should_compact,
        compact_history_fn=compact_history,
        emit_trace=emit_trace,
    )

    emit_trace("run_analyst_turn.build_deps", "요청 컨텍스트 의존성을 구성합니다.")
    deps = _build_deps(payload)
    deps.trace_callback = emit_trace
    emit_trace(
        "run_analyst_turn.history",
        (
            "모델 히스토리를 구성했습니다. "
            f"message_history={len(prepared_context.history)}, source={prepared_context.memory_source}"
        ),
    )

    try:
        emit_trace("run_analyst_turn.agent_run_sync", "Agent.run_sync를 호출합니다.")
        run_result = _get_agent().run_sync(
            user_prompt=text,
            deps=deps,
            message_history=prepared_context.history,
            usage_limits=config.DEFAULT_USAGE_LIMITS,
        )
        emit_trace("run_analyst_turn.agent_run_sync", "Agent.run_sync 호출이 완료되었습니다.")

        run_messages_json = run_result.all_messages_json().decode("utf-8")
        result = result_contract.apply_output_to_result(
            output=run_result.output,
            deps=deps,
            run_messages_json=run_messages_json,
            payload=payload,
            save_approval_context=_save_approval_context,
        )
        emit_trace(
            "run_analyst_turn.apply_output",
            f"모델 출력을 응답 포맷으로 변환했습니다. action={result.get('action')}",
        )

        prompt_usage: int | None = None
        try:
            prompt_usage = int(run_result.usage().input_tokens)
        except Exception:
            prompt_usage = None

        if prepared_context.thread_store is not None and prepared_context.thread_key is not None:
            try:
                append_result = thread_context.append_turn_to_thread_memory(
                    thread_store=prepared_context.thread_store,
                    thread_key=prepared_context.thread_key,
                    run_result=run_result,
                    run_messages_json=run_messages_json,
                    prompt_usage=prompt_usage,
                    previous_prompt_tokens_ema=(
                        prepared_context.thread_state.prompt_tokens_ema
                        if prepared_context.thread_state is not None
                        else None
                    ),
                    update_prompt_tokens_ema_fn=thread_context.update_prompt_tokens_ema,
                )
                emit_trace(
                    "run_analyst_turn.memory_save",
                    (
                        "DynamoDB thread memory에 턴을 저장했습니다. "
                        f"turn={append_result.turn}, prompt_usage={prompt_usage}"
                    ),
                )
                result["memory_backend"] = "dynamodb"
            except Exception as exc:
                emit_trace("run_analyst_turn.memory_save", f"턴 저장 실패: {exc}")

        result["memory_source"] = prepared_context.memory_source
        return _with_trace(result, trace)
    except Exception as exc:
        recovered = recovery.recover_from_tool_retry_error(exc, deps)
        if recovered is not None:
            emit_trace(
                "run_analyst_turn.recover",
                "도구 재시도 한도 오류를 복구 경로로 처리했습니다.",
            )
            return _with_trace(recovered, trace)

        recovered_request_limit = recovery.recover_from_request_limit_error(exc, deps)
        if recovered_request_limit is not None:
            emit_trace(
                "run_analyst_turn.recover",
                "request_limit 초과 오류를 복구 경로로 처리했습니다.",
            )
            return _with_trace(recovered_request_limit, trace)

        emit_trace("run_analyst_turn.error", f"오류로 실패했습니다: {exc}")
        result = {
            **_base_result(),
            "action": "error",
            "error": str(exc),
            "response_text": f":x: 요청 처리 중 오류가 발생했습니다.\n{exc}",
        }
        return _with_trace(result, trace)


def run_analyst_approval(payload: dict[str, Any]) -> dict[str, Any]:
    """Resume a deferred approval request and continue execution."""
    return approval_flow.run_approval_flow(
        payload,
        approval_store_factory=lambda: DynamoApprovalStore.from_env(),
        build_deps=_build_deps,
        get_agent=_get_agent,
        usage_limits=config.DEFAULT_USAGE_LIMITS,
        apply_output_to_result=result_contract.apply_output_to_result,
        base_result=result_contract.base_result,
        recover_from_tool_retry_error=recovery.recover_from_tool_retry_error,
        thread_memory_enabled=thread_context.thread_memory_enabled,
        thread_key_from_payload=thread_context.thread_key_from_payload,
        thread_store_factory=lambda: DynamoThreadMemoryStore.from_env(),
    )
