import pytest
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel

from data_bolt.tasks.analyst_agent import model_factory


def test_build_model_openai_compatible(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("LLM_OPENAI_MODEL", "gpt-4o-mini")

    model = model_factory.build_model_for_env()

    assert isinstance(model, OpenAIChatModel)


def test_build_model_laas(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "laas")

    model = model_factory.build_model_for_env()

    assert isinstance(model, FunctionModel)


def test_build_model_invalid_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "unknown")

    with pytest.raises(ValueError):
        model_factory.build_model_for_env()


def test_parse_laas_response_tool_calls() -> None:
    parsed = model_factory._parse_laas_response_to_model_response(
        {
            "id": "resp-1",
            "model": "laas-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "bigquery_dry_run",
                                    "arguments": '{"sql":"SELECT 1"}',
                                },
                            }
                        ],
                    },
                }
            ],
        }
    )

    assert parsed.provider_name == "laas"
    assert parsed.parts


def test_parse_laas_response_tool_calls_from_content_json() -> None:
    parsed = model_factory._parse_laas_response_to_model_response(
        {
            "id": "resp-2",
            "model": "laas-model",
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"tool_calls":[{"id":"call-2","type":"function","function":'
                            '{"name":"bigquery_execute","arguments":"{\\"sql\\":\\"SELECT 1\\"}"}}]}'
                        )
                    },
                }
            ],
        }
    )

    assert parsed.parts
    first_part = parsed.parts[0]
    assert isinstance(first_part, ToolCallPart)
    assert first_part.tool_name == "bigquery_execute"
    assert first_part.args_as_dict().get("sql") == "SELECT 1"


def test_parse_laas_response_normalizes_get_schema_context_args() -> None:
    parsed = model_factory._parse_laas_response_to_model_response(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"tool_call":{"tool_name":"get_schema_context","parameters":'
                            '{"query":"가입자 테이블 알려줘","top_k":"5"}}}'
                        )
                    }
                }
            ]
        }
    )

    first_part = parsed.parts[0]
    assert isinstance(first_part, ToolCallPart)
    assert first_part.tool_name == "get_schema_context"
    args = first_part.args_as_dict()
    assert args.get("question") == "가입자 테이블 알려줘"
    assert args.get("top_k") == 5
