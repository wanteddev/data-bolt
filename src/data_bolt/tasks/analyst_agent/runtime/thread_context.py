"""Thread memory loading/compaction helpers."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
)

from ..semantic_compaction import CompactionConfig, build_memory_message
from ..thread_memory_store import AppendTurnResult, ThreadKey, ThreadMemoryState

type TraceEmitter = Callable[[str, str], None]


@dataclass
class PreparedThreadContext:
    history: list[ModelMessage]
    memory_source: str
    thread_key: ThreadKey | None
    thread_state: ThreadMemoryState | None
    thread_store: Any | None


def thread_memory_backend() -> str:
    return (os.getenv("BIGQUERY_THREAD_MEMORY_BACKEND") or "memory").strip().lower()


def thread_memory_enabled() -> bool:
    return thread_memory_backend() == "dynamodb"


def thread_key_from_payload(payload: Mapping[str, Any]) -> ThreadKey | None:
    key = ThreadKey(
        team_id=str(payload.get("team_id") or ""),
        channel_id=str(payload.get("channel_id") or ""),
        thread_ts=str(payload.get("thread_ts") or ""),
    )
    return key if key.is_valid else None


def history_to_model_messages(history: Any) -> list[ModelMessage]:
    if not isinstance(history, list):
        return []

    messages: list[ModelMessage] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip().lower()
        content = str(item.get("content") or "").strip()
        if not content:
            continue

        if role == "assistant":
            messages.append(ModelResponse(parts=[TextPart(content=content)]))
        else:
            messages.append(ModelRequest.user_text_prompt(content))
    return messages


def parse_model_messages_json(raw_json: str) -> list[ModelMessage]:
    text = raw_json.strip() or "[]"
    try:
        return ModelMessagesTypeAdapter.validate_json(text.encode("utf-8"))
    except Exception:
        return []


def build_history_from_thread_state(
    state: ThreadMemoryState,
    config: CompactionConfig,
) -> list[ModelMessage]:
    if not state.turns:
        summary = state.summary
        if summary is None:
            return []
        memory_message = build_memory_message(summary.summary_text, summary.summary_struct_json)
        return [memory_message] if memory_message is not None else []

    seed_turn = next((turn for turn in state.turns if turn.turn == 1), state.turns[0])
    recent_start_turn = max(
        state.summary_last_turn + 1, state.latest_turn - config.keep_recent_turns + 1
    )

    history: list[ModelMessage] = []
    history.extend(parse_model_messages_json(seed_turn.new_messages_json))
    if state.summary is not None:
        memory_message = build_memory_message(
            state.summary.summary_text, state.summary.summary_struct_json
        )
        if memory_message is not None:
            history.append(memory_message)

    for turn in state.turns:
        if turn.turn == seed_turn.turn:
            continue
        if turn.turn >= recent_start_turn:
            history.extend(parse_model_messages_json(turn.new_messages_json))
    return history


def compaction_candidates(
    state: ThreadMemoryState,
    config: CompactionConfig,
) -> tuple[list[ModelMessage], int, int]:
    if state.latest_turn <= 1:
        return [], 0, 0

    source_turn_start = max(state.summary_last_turn + 1, 2)
    protected_recent_start = max(2, state.latest_turn - config.keep_recent_turns + 1)
    source_turn_end = protected_recent_start - 1
    if source_turn_end < source_turn_start:
        return [], 0, 0

    selected = [turn for turn in state.turns if source_turn_start <= turn.turn <= source_turn_end]
    messages: list[ModelMessage] = []
    for turn in selected:
        messages.extend(parse_model_messages_json(turn.new_messages_json))
    return messages, source_turn_start, source_turn_end


def update_prompt_tokens_ema(
    previous: float | None, current_prompt_tokens: int | None
) -> float | None:
    if current_prompt_tokens is None or current_prompt_tokens <= 0:
        return previous
    if previous is None or previous <= 0:
        return float(current_prompt_tokens)
    return (previous * 0.7) + (float(current_prompt_tokens) * 0.3)


def has_thread_memory(
    payload: Mapping[str, Any],
    *,
    thread_store_factory: Callable[[], Any],
) -> bool:
    if not thread_memory_enabled():
        return False
    key = thread_key_from_payload(payload)
    if key is None:
        return False
    try:
        return bool(thread_store_factory().has_state(key))
    except Exception:
        return False


def prepare_thread_context(
    *,
    payload: Mapping[str, Any],
    history_raw: Any,
    compaction_config: CompactionConfig,
    thread_store_factory: Callable[[], Any],
    emit_trace: TraceEmitter | None = None,
) -> PreparedThreadContext:
    history = history_to_model_messages(history_raw)
    memory_source = "payload"
    thread_key = thread_key_from_payload(payload)
    thread_state: ThreadMemoryState | None = None
    thread_store: Any | None = None

    if thread_memory_enabled() and thread_key is not None:
        try:
            store = thread_store_factory()
            thread_state = store.load_state(thread_key)
            thread_store = store
            if thread_state is not None:
                history = build_history_from_thread_state(thread_state, compaction_config)
                memory_source = "dynamodb"
                if emit_trace is not None:
                    emit_trace(
                        "run_analyst_turn.memory_load",
                        (
                            "DynamoDB thread memory를 로드했습니다. "
                            f"latest_turn={thread_state.latest_turn}, "
                            f"summary_version={thread_state.summary_version}"
                        ),
                    )
            else:
                memory_source = "payload" if history else "none"
                if emit_trace is not None:
                    emit_trace("run_analyst_turn.memory_load", "DynamoDB memory miss")
        except Exception as exc:
            if emit_trace is not None:
                emit_trace("run_analyst_turn.memory_load", f"DynamoDB memory load 실패: {exc}")
            thread_store = None
            thread_state = None

    return PreparedThreadContext(
        history=history,
        memory_source=memory_source,
        thread_key=thread_key,
        thread_state=thread_state,
        thread_store=thread_store,
    )


def maybe_compact_thread_context(
    *,
    context: PreparedThreadContext,
    text: str,
    compaction_config: CompactionConfig,
    estimate_tokens_fn: Callable[..., int],
    should_compact_fn: Callable[..., bool],
    compact_history_fn: Callable[..., Any],
    emit_trace: TraceEmitter,
) -> tuple[PreparedThreadContext, int]:
    estimated_tokens = estimate_tokens_fn(
        messages=context.history,
        user_prompt=text,
        prompt_tokens_ema=(
            context.thread_state.prompt_tokens_ema if context.thread_state is not None else None
        ),
    )
    emit_trace(
        "run_analyst_turn.token_estimate",
        f"estimated_prompt_tokens={estimated_tokens}, source={context.memory_source}",
    )

    if (
        context.thread_store is None
        or context.thread_key is None
        or context.thread_state is None
        or not should_compact_fn(
            estimated_prompt_tokens=estimated_tokens,
            config=compaction_config,
        )
    ):
        return context, estimated_tokens

    emit_trace("run_analyst_turn.compaction", "semantic compaction 조건을 충족했습니다.")
    current_state = context.thread_state
    for attempt in range(2):
        candidates, source_turn_start, source_turn_end = compaction_candidates(
            current_state, compaction_config
        )
        if not candidates:
            emit_trace(
                "run_analyst_turn.compaction",
                "압축 대상 턴이 없어 compaction을 건너뜁니다.",
            )
            break

        existing_summary = current_state.summary
        rebase_every = max(compaction_config.rebase_every, 1)
        rebase_mode = (
            current_state.summary_version > 0
            and (current_state.summary_version + 1) % rebase_every == 0
        )

        try:
            compacted = compact_history_fn(
                existing_summary_text=(
                    existing_summary.summary_text if existing_summary is not None else ""
                ),
                existing_summary_struct_json=(
                    existing_summary.summary_struct_json if existing_summary is not None else "{}"
                ),
                candidate_messages=candidates,
                max_summary_chars=compaction_config.max_summary_chars,
                rebase_mode=rebase_mode,
            )
        except Exception as exc:
            emit_trace("run_analyst_turn.compaction", f"compaction 실패: {exc}")
            if compaction_config.fail_open:
                break
            raise

        saved = context.thread_store.save_summary_with_cas(
            key=context.thread_key,
            expected_version=current_state.version,
            expected_summary_version=current_state.summary_version,
            summary_text=compacted.summary_text,
            summary_struct_json=compacted.summary_struct_json,
            source_turn_start=source_turn_start,
            source_turn_end=source_turn_end,
            prompt_tokens_ema=current_state.prompt_tokens_ema,
        )

        if saved:
            refreshed = context.thread_store.load_state(context.thread_key)
            if refreshed is not None:
                current_state = refreshed
                context.history = build_history_from_thread_state(current_state, compaction_config)
                context.memory_source = "dynamodb_compacted"
            emit_trace(
                "run_analyst_turn.compaction",
                f"semantic compaction 적용 완료. source_turns={source_turn_start}-{source_turn_end}",
            )
            break

        if attempt == 0:
            emit_trace(
                "run_analyst_turn.compaction",
                "CAS 충돌로 최신 상태를 재조회 후 1회 재시도합니다.",
            )
            refreshed = context.thread_store.load_state(context.thread_key)
            if refreshed is None:
                break
            current_state = refreshed
            continue

        emit_trace("run_analyst_turn.compaction", "CAS 재시도 실패")

    context.thread_state = current_state
    return context, estimated_tokens


def append_turn_to_thread_memory(
    *,
    thread_store: Any,
    thread_key: ThreadKey,
    run_result: Any,
    run_messages_json: str,
    prompt_usage: int | None,
    previous_prompt_tokens_ema: float | None,
    update_prompt_tokens_ema_fn: Callable[[float | None, int | None], float | None],
) -> AppendTurnResult:
    next_ema = update_prompt_tokens_ema_fn(previous_prompt_tokens_ema, prompt_usage)
    try:
        new_messages_json = run_result.new_messages_json().decode("utf-8")
    except Exception:
        new_messages_json = run_messages_json
    return thread_store.append_turn(
        key=thread_key,
        new_messages_json=new_messages_json,
        prompt_tokens_ema=next_ema,
        usage_prompt_tokens=prompt_usage,
    )
