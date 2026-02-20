from pydantic_ai.messages import ModelRequest

from data_bolt.tasks.analyst_agent.semantic_compaction import (
    build_memory_message,
    estimate_tokens,
    load_compaction_config,
    should_compact,
)


def test_estimate_tokens_prefers_prompt_tokens_ema() -> None:
    estimate = estimate_tokens(messages=[], user_prompt="안녕하세요", prompt_tokens_ema=321.7)
    assert estimate == 321


def test_should_compact_uses_trigger_ratio() -> None:
    config = load_compaction_config()
    assert should_compact(estimated_prompt_tokens=90000, config=config) is True
    assert should_compact(estimated_prompt_tokens=1000, config=config) is False


def test_build_memory_message_returns_system_prompt() -> None:
    message = build_memory_message(
        "사용자 이름은 우징",
        '{"durable_facts":["사용자 이름은 우징"],"summary_text":"사용자 이름은 우징"}',
    )
    assert isinstance(message, ModelRequest)
    payload = str(message.parts[0].content)
    assert "사용자 이름은 우징" in payload
