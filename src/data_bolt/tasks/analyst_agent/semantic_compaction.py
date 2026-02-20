"""Semantic compaction helpers for thread memory history."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, cast

from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    SystemPromptPart,
)

from .model_factory import build_model_for_env


@dataclass(frozen=True)
class CompactionConfig:
    """Runtime configuration for semantic compaction."""

    context_window_tokens: int
    trigger_ratio: float
    target_ratio: float
    reserve_output_tokens: int
    keep_recent_turns: int
    max_summary_chars: int
    rebase_every: int
    fail_open: bool


@dataclass(frozen=True)
class CompactionResult:
    """Parsed compaction output."""

    summary_text: str
    summary_struct: dict[str, Any]

    @property
    def summary_struct_json(self) -> str:
        return json.dumps(self.summary_struct, ensure_ascii=False)


class CompactionSummarySchema(BaseModel):
    durable_facts: list[str]
    open_tasks: list[str]
    user_preferences: list[str]
    sql_artifacts: list[str]
    constraints: list[str]
    risk_flags: list[str]
    summary_text: str


_summary_agent: Agent[None, Any] | None = None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def load_compaction_config() -> CompactionConfig:
    """Load compaction-related settings from env."""
    return CompactionConfig(
        context_window_tokens=_env_int("BIGQUERY_CONTEXT_WINDOW_TOKENS", 128000),
        trigger_ratio=_env_float("BIGQUERY_CONTEXT_TRIGGER_RATIO", 0.70),
        target_ratio=_env_float("BIGQUERY_CONTEXT_TARGET_RATIO", 0.50),
        reserve_output_tokens=_env_int("BIGQUERY_CONTEXT_RESERVE_OUTPUT_TOKENS", 4000),
        keep_recent_turns=_env_int("BIGQUERY_CONTEXT_KEEP_RECENT_TURNS", 8),
        max_summary_chars=_env_int("BIGQUERY_CONTEXT_MAX_SUMMARY_CHARS", 4000),
        rebase_every=_env_int("BIGQUERY_CONTEXT_REBASE_EVERY", 8),
        fail_open=_env_bool("BIGQUERY_CONTEXT_FAIL_OPEN", True),
    )


def _messages_to_string(messages: list[ModelMessage], user_prompt: str) -> str:
    history_json = ModelMessagesTypeAdapter.dump_json(messages).decode("utf-8")
    return f"{history_json}\n\n[USER_PROMPT]\n{user_prompt}"


def estimate_tokens(
    *,
    messages: list[ModelMessage],
    user_prompt: str,
    prompt_tokens_ema: float | None,
) -> int:
    """Estimate prompt tokens with EMA > tokenizer > utf-8 fallback priority."""
    if isinstance(prompt_tokens_ema, (int, float)) and prompt_tokens_ema > 0:
        return int(prompt_tokens_ema)

    source = _messages_to_string(messages, user_prompt)
    provider = (os.getenv("LLM_PROVIDER") or "laas").strip().lower()
    if provider == "openai_compatible":
        try:
            import tiktoken

            model_name = (os.getenv("LLM_OPENAI_MODEL") or "gpt-4o-mini").strip()
            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(source))
        except Exception:
            pass

    # Conservative fallback for mixed Korean/English payloads.
    return max(1, len(source.encode("utf-8")) // 3)


def should_compact(*, estimated_prompt_tokens: int, config: CompactionConfig) -> bool:
    """Return whether semantic compaction is required."""
    threshold = int(config.context_window_tokens * config.trigger_ratio)
    return estimated_prompt_tokens + config.reserve_output_tokens > threshold


def build_memory_message(summary_text: str, summary_struct_json: str) -> ModelMessage | None:
    """Build a system-style memory message from summary text/structure."""
    trimmed = summary_text.strip()
    if not trimmed:
        return None
    content = (
        "Conversation memory summary (compressed context).\n"
        "Treat this as durable memory, but prioritize more recent raw turns if conflicts exist.\n"
        f"[SUMMARY]\n{trimmed}\n"
        f"[SUMMARY_JSON]\n{summary_struct_json.strip() or '{}'}"
    )
    return ModelRequest(parts=[SystemPromptPart(content=content)])


def _get_summary_agent() -> Agent[None, Any]:
    global _summary_agent
    if _summary_agent is None:
        _summary_agent = Agent(
            build_model_for_env(),
            output_type=str,
            instructions=(
                "You compress conversation history for a production Slack analyst bot.\n"
                "Return ONE valid JSON object only.\n"
                "Required keys:\n"
                "- durable_facts: list[str]\n"
                "- open_tasks: list[str]\n"
                "- user_preferences: list[str]\n"
                "- sql_artifacts: list[str]\n"
                "- constraints: list[str]\n"
                "- risk_flags: list[str]\n"
                "- summary_text: str\n"
                "Rules:\n"
                "- Separate factual statements from uncertain claims.\n"
                "- Mark uncertainty as 'unknown'.\n"
                "- Preserve user identity/preferences and unresolved tasks.\n"
                "- Keep latest SQL decisions and mark discarded drafts.\n"
            ),
            output_retries=1,
            defer_model_check=True,
        )
    return _summary_agent


def _ensure_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _parse_compaction_result(raw_text: str, max_summary_chars: int) -> CompactionResult:
    parsed: dict[str, Any] = {}
    try:
        candidate = json.loads(raw_text)
        if isinstance(candidate, dict):
            parsed = candidate
    except Exception:
        parsed = {}
    if not parsed:
        raise ValueError("Compaction output is not valid JSON object")

    normalized = {
        "durable_facts": _ensure_string_list(parsed.get("durable_facts")),
        "open_tasks": _ensure_string_list(parsed.get("open_tasks")),
        "user_preferences": _ensure_string_list(parsed.get("user_preferences")),
        "sql_artifacts": _ensure_string_list(parsed.get("sql_artifacts")),
        "constraints": _ensure_string_list(parsed.get("constraints")),
        "risk_flags": _ensure_string_list(parsed.get("risk_flags")),
        "summary_text": str(parsed.get("summary_text") or "").strip(),
    }

    try:
        validated = CompactionSummarySchema.model_validate(normalized)
    except ValidationError as exc:
        raise ValueError(f"Compaction output schema mismatch: {exc}") from exc

    summary_text = validated.summary_text[:max_summary_chars].strip()
    if not summary_text:
        summary_text = "unknown"

    summary_struct = {
        "durable_facts": validated.durable_facts,
        "open_tasks": validated.open_tasks,
        "user_preferences": validated.user_preferences,
        "sql_artifacts": validated.sql_artifacts,
        "constraints": validated.constraints,
        "risk_flags": validated.risk_flags,
        "summary_text": summary_text,
    }
    return CompactionResult(summary_text=summary_text, summary_struct=summary_struct)


def compact_history(
    *,
    existing_summary_text: str,
    existing_summary_struct_json: str,
    candidate_messages: list[ModelMessage],
    max_summary_chars: int,
    rebase_mode: bool,
) -> CompactionResult:
    """Run semantic compaction with the same model provider as the main agent."""
    payload = {
        "existing_summary_text": existing_summary_text.strip(),
        "existing_summary_struct_json": existing_summary_struct_json.strip() or "{}",
        "rebase_mode": rebase_mode,
        "candidate_messages_json": ModelMessagesTypeAdapter.dump_json(candidate_messages).decode(
            "utf-8"
        ),
        "output_contract": {
            "durable_facts": ["..."],
            "open_tasks": ["..."],
            "user_preferences": ["..."],
            "sql_artifacts": ["..."],
            "constraints": ["..."],
            "risk_flags": ["..."],
            "summary_text": "...",
        },
    }

    prompt = (
        "Compress the conversation history into durable memory.\n"
        "Return one JSON object only.\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    result = _get_summary_agent().run_sync(user_prompt=prompt)
    output_text = str(cast(Any, result).output or "")
    return _parse_compaction_result(output_text, max_summary_chars=max_summary_chars)
