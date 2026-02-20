"""Deferred approval resume flow helpers."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from typing import Any, cast

from pydantic_ai import DeferredToolRequests, DeferredToolResults, UsageLimits
from pydantic_ai.messages import ModelMessagesTypeAdapter

from ..approval_store import ApprovalContext, DynamoApprovalStore, build_approval_context_timestamps
from ..deps import AnalystDeps
from ..thread_memory_store import ThreadKey
from .thread_context import append_turn_to_thread_memory, update_prompt_tokens_ema

type BuildDeps = Callable[[Mapping[str, Any]], AnalystDeps]
type GetAgent = Callable[[], Any]
type ApplyOutputToResult = Callable[..., dict[str, Any]]
type RecoverFromToolRetryError = Callable[[Exception, AnalystDeps], dict[str, Any] | None]
type BaseResultFactory = Callable[[], dict[str, Any]]
type ThreadMemoryEnabled = Callable[[], bool]
type ThreadKeyBuilder = Callable[[Mapping[str, Any]], ThreadKey | None]


def persist_deferred_approval_context(
    *,
    deferred: DeferredToolRequests,
    payload: Mapping[str, Any],
    run_messages_json: str,
    approval_store: DynamoApprovalStore,
) -> str:
    approval_request_id = str(uuid.uuid4())
    created_at, expires_at = build_approval_context_timestamps()

    tool_call_ids = [call.tool_call_id for call in deferred.approvals]
    context = ApprovalContext(
        approval_request_id=approval_request_id,
        requester_user_id=str(payload.get("user_id") or ""),
        channel_id=str(payload.get("channel_id") or ""),
        thread_ts=str(payload.get("thread_ts") or ""),
        team_id=str(payload.get("team_id") or ""),
        tool_call_ids=tool_call_ids,
        deferred_metadata=deferred.metadata,
        session_messages_json=run_messages_json,
        created_at=created_at,
        expires_at=expires_at,
    )
    approval_store.save(context)
    return approval_request_id


def run_approval_flow(
    payload: dict[str, Any],
    *,
    approval_store_factory: Callable[[], DynamoApprovalStore],
    build_deps: BuildDeps,
    get_agent: GetAgent,
    usage_limits: UsageLimits,
    apply_output_to_result: ApplyOutputToResult,
    base_result: BaseResultFactory,
    recover_from_tool_retry_error: RecoverFromToolRetryError,
    thread_memory_enabled: ThreadMemoryEnabled,
    thread_key_from_payload: ThreadKeyBuilder,
    thread_store_factory: Callable[[], Any],
) -> dict[str, Any]:
    approval_request_id = str(payload.get("approval_request_id") or "").strip()
    approved = bool(payload.get("approved"))
    actor_user_id = str(payload.get("user_id") or "")

    if not approval_request_id:
        return {
            **base_result(),
            "action": "error",
            "error": "approval_request_id is required",
            "response_text": ":x: 승인 요청 식별자가 없습니다.",
        }

    deps: AnalystDeps | None = None
    try:
        store = approval_store_factory()
        context = store.load(approval_request_id)
        if context is None:
            return {
                **base_result(),
                "action": "error",
                "error": "approval request not found",
                "response_text": ":warning: 승인 대기 요청을 찾을 수 없습니다. 다시 요청해주세요.",
            }

        if (
            actor_user_id
            and context.requester_user_id
            and actor_user_id != context.requester_user_id
        ):
            return {
                **base_result(),
                "action": "error",
                "error": "permission denied",
                "response_text": ":x: 원 요청자만 승인/취소할 수 있습니다.",
                "should_respond": True,
            }

        approvals = {tool_call_id: approved for tool_call_id in context.tool_call_ids}
        deferred_results = DeferredToolResults(
            approvals=cast(Any, approvals),
            metadata=context.deferred_metadata,
        )
        message_history = ModelMessagesTypeAdapter.validate_json(
            context.session_messages_json.encode("utf-8")
        )

        run_payload = {
            "user_id": context.requester_user_id,
            "channel_id": context.channel_id,
            "thread_ts": context.thread_ts,
            "team_id": context.team_id,
        }
        deps = build_deps(run_payload)

        run_result = get_agent().run_sync(
            deps=deps,
            message_history=message_history,
            deferred_tool_results=deferred_results,
            usage_limits=usage_limits,
        )

        store.delete(approval_request_id)
        run_messages_json = run_result.all_messages_json().decode("utf-8")
        result = apply_output_to_result(
            output=run_result.output,
            deps=deps,
            run_messages_json=run_messages_json,
            payload=run_payload,
            save_approval_context=store.save,
        )

        thread_key = thread_key_from_payload(run_payload)
        if thread_memory_enabled() and thread_key is not None:
            try:
                memory_store = thread_store_factory()
                prior_state = memory_store.load_state(thread_key)
                prompt_usage: int | None = None
                try:
                    prompt_usage = int(run_result.usage().input_tokens)
                except Exception:
                    prompt_usage = None
                append_turn_to_thread_memory(
                    thread_store=memory_store,
                    thread_key=thread_key,
                    run_result=run_result,
                    run_messages_json=run_messages_json,
                    prompt_usage=prompt_usage,
                    previous_prompt_tokens_ema=(
                        prior_state.prompt_tokens_ema if prior_state is not None else None
                    ),
                    update_prompt_tokens_ema_fn=update_prompt_tokens_ema,
                )
                result["memory_backend"] = "dynamodb"
            except Exception:
                pass

        # Keep a newly-issued approval_request_id when follow-up approval is required.
        if result.get("action") != "approval_required":
            result["approval_request_id"] = approval_request_id
        result["approved"] = approved
        return result

    except Exception as exc:
        if deps is not None:
            recovered = recover_from_tool_retry_error(exc, deps)
            if recovered is not None:
                return recovered
        return {
            **base_result(),
            "action": "error",
            "error": str(exc),
            "response_text": f":x: 승인 처리 중 오류가 발생했습니다.\n{exc}",
        }
