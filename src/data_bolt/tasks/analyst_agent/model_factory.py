"""LLM model factory for analyst agent (OpenAI-compatible + LAAS)."""

from __future__ import annotations

import json
import os
from typing import Any, cast

import httpx
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RequestUsage

LAAS_DEFAULT_BASE_URL = "https://api-laas.wanted.co.kr"
LAAS_DEFAULT_PRESET_HASH = "2e1cfa82b035c26cbbbdae632cea070514eb8b773f616aaeaf668e2f0be8f10d"


def _read_timeout_seconds() -> float:
    raw = os.getenv("LLM_TIMEOUT_GENERATION_SECONDS") or os.getenv("LLM_TIMEOUT_SECONDS") or "120"
    try:
        timeout = float(raw)
    except ValueError:
        timeout = 120.0
    return timeout if timeout > 0 else 120.0


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_extract_text_content(item) for item in content]
        return "\n".join(part for part in parts if part.strip())
    if isinstance(content, dict):
        for key in ("text", "content", "value", "output_text"):
            if key in content:
                text = _extract_text_content(content[key])
                if text.strip():
                    return text
    return ""


def _parse_json_object(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _to_laas_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    payload_messages: list[dict[str, Any]] = []

    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    payload_messages.append({"role": "system", "content": part.content})
                elif isinstance(part, UserPromptPart):
                    payload_messages.append(
                        {
                            "role": "user",
                            "content": _extract_text_content(part.content),
                        }
                    )
                elif isinstance(part, ToolReturnPart):
                    payload_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Tool result ({part.tool_name}): "
                                + (
                                    part.content
                                    if isinstance(part.content, str)
                                    else json.dumps(part.content, ensure_ascii=False, default=str)
                                )
                            ),
                        }
                    )
                else:
                    payload_messages.append(
                        {
                            "role": "user",
                            "content": (
                                part.content
                                if isinstance(part.content, str)
                                else json.dumps(part.content, ensure_ascii=False, default=str)
                            ),
                        }
                    )
        else:
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for part in message.parts:
                if isinstance(part, TextPart):
                    text_parts.append(part.content)
                elif isinstance(part, ToolCallPart):
                    tool_calls.append(
                        {
                            "id": part.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": part.tool_name,
                                "arguments": part.args_as_json_str(),
                            },
                        }
                    )

            if text_parts or tool_calls:
                assistant_text = "\n".join(text_parts).strip() if text_parts else ""
                if tool_calls:
                    tool_call_text = json.dumps({"tool_calls": tool_calls}, ensure_ascii=False)
                    assistant_text = (
                        f"{assistant_text}\n{tool_call_text}".strip()
                        if assistant_text
                        else tool_call_text
                    )
                payload_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": assistant_text,
                }
                payload_messages.append(payload_message)

    return payload_messages


def _tool_defs_to_openai_tools(info: AgentInfo) -> list[dict[str, Any]]:
    tool_defs = [*info.function_tools, *info.output_tools]
    tools: list[dict[str, Any]] = []
    for tool in tool_defs:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.parameters_json_schema,
                },
            }
        )
    return tools


def _parse_laas_response_to_model_response(data: Any) -> ModelResponse:
    response_dict: dict[str, Any] = data if isinstance(data, dict) else {}
    choices = response_dict.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else None
    choice: dict[str, Any] = first_choice if isinstance(first_choice, dict) else {}

    message_raw = choice.get("message")
    message: dict[str, Any] = message_raw if isinstance(message_raw, dict) else {}
    parts: list[TextPart | ToolCallPart] = []

    def _normalize_tool_args(tool_name: str, args: str | dict[str, Any]) -> str | dict[str, Any]:
        if isinstance(args, dict):
            args_dict = dict(args)
        else:
            try:
                parsed = json.loads(args)
            except Exception:
                return args
            if isinstance(parsed, dict):
                args_dict = parsed
            else:
                return args

        if tool_name in {"bigquery_dry_run", "bigquery_execute"} and "sql" not in args_dict:
            for key in ("query", "query_text", "statement", "text"):
                candidate = args_dict.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    args_dict["sql"] = candidate
                    break

        if tool_name == "get_schema_context":
            if "question" not in args_dict:
                for key in ("query", "text", "prompt", "question_text"):
                    candidate = args_dict.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        args_dict["question"] = candidate
                        break
                else:
                    sql_candidate = args_dict.get("sql")
                    if isinstance(sql_candidate, str) and sql_candidate.strip():
                        args_dict["question"] = sql_candidate
            top_k = args_dict.get("top_k")
            if isinstance(top_k, str):
                try:
                    args_dict["top_k"] = int(top_k)
                except ValueError:
                    pass

        return args_dict

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            raw_function = call.get("function")
            function = raw_function if isinstance(raw_function, dict) else {}
            tool_name = function.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                continue
            raw_args = function.get("arguments")
            args_from_function: str | dict[str, Any] | None
            if isinstance(raw_args, str):
                args_from_function = raw_args
            elif isinstance(raw_args, dict):
                args_from_function = raw_args
            else:
                args_from_function = "{}"
            tool_call_id = call.get("id") if isinstance(call.get("id"), str) else ""
            normalized_args = _normalize_tool_args(tool_name, args_from_function)
            parts.append(
                ToolCallPart(
                    tool_name=tool_name,
                    args=normalized_args,
                    tool_call_id=tool_call_id or f"pyd_ai_tool_call_id__{tool_name}",
                )
            )

    content = _extract_text_content(message.get("content"))
    parsed_json = _parse_json_object(content)

    def _append_tool_call_part_from_json_call(call: dict[str, Any]) -> None:
        raw_function = call.get("function")
        function = raw_function if isinstance(raw_function, dict) else {}

        tool_name_raw = function.get("name") or call.get("tool_name") or call.get("name")
        tool_name = tool_name_raw if isinstance(tool_name_raw, str) else ""
        if not tool_name:
            return

        args_candidate = function.get("arguments")
        if args_candidate is None:
            args_candidate = call.get("parameters")
        if args_candidate is None:
            args_candidate = call.get("args")

        args: str | dict[str, Any]
        if isinstance(args_candidate, (str, dict)):
            args = _normalize_tool_args(tool_name, args_candidate)
        else:
            args = "{}"

        tool_call_id_raw = call.get("tool_call_id") or call.get("id")
        tool_call_id = (
            str(tool_call_id_raw)
            if isinstance(tool_call_id_raw, (str, int))
            else f"pyd_ai_tool_call_id__{tool_name}"
        )

        parts.append(ToolCallPart(tool_name=tool_name, args=args, tool_call_id=tool_call_id))

    raw_tool_calls = parsed_json.get("tool_calls")
    if isinstance(raw_tool_calls, list):
        for call in raw_tool_calls:
            if not isinstance(call, dict):
                continue
            _append_tool_call_part_from_json_call(call)

    raw_tool_call = parsed_json.get("tool_call")
    if isinstance(raw_tool_call, dict):
        _append_tool_call_part_from_json_call(raw_tool_call)

    if not parts:
        parts.append(TextPart(content=content or ""))

    raw_usage = response_dict.get("usage")
    usage_raw = raw_usage if isinstance(raw_usage, dict) else {}
    prompt_tokens = usage_raw.get("prompt_tokens")
    completion_tokens = usage_raw.get("completion_tokens")

    usage = RequestUsage(
        input_tokens=int(prompt_tokens) if isinstance(prompt_tokens, int) else 0,
        output_tokens=int(completion_tokens) if isinstance(completion_tokens, int) else 0,
    )

    return ModelResponse(
        parts=parts,
        usage=usage,
        model_name=(
            str(response_dict.get("model")) if isinstance(response_dict.get("model"), str) else None
        ),
        provider_name="laas",
        provider_url=os.getenv("LAAS_BASE_URL", LAAS_DEFAULT_BASE_URL).rstrip("/"),
        finish_reason=(
            choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None
        ),
        provider_response_id=(
            str(response_dict.get("id")) if isinstance(response_dict.get("id"), str) else None
        ),
    )


def _laas_function_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    api_key = os.getenv("LAAS_API_KEY", "").strip()
    if not api_key:
        raise ValueError("LAAS_API_KEY is required when LLM_PROVIDER=laas")

    base_url = os.getenv("LAAS_BASE_URL", LAAS_DEFAULT_BASE_URL).rstrip("/")
    preset_hash = os.getenv("LAAS_EMPTY_PRESET_HASH", LAAS_DEFAULT_PRESET_HASH)
    request_messages = _to_laas_messages(messages)
    tools = _tool_defs_to_openai_tools(info)
    tool_names = [str(tool["function"]["name"]) for tool in tools]
    laas_bridge_prompt = (
        "json mode is enabled. Always respond with valid JSON only. "
        "If you need a tool, respond with "
        '{"tool_call":{"tool_name":"<name>","parameters":{...}}}. '
        "When no tool is needed, return JSON matching the expected response schema. "
        f"Available tools: {', '.join(tool_names) if tool_names else 'none'}."
    )
    request_messages = [{"role": "system", "content": laas_bridge_prompt}, *request_messages]

    payload: dict[str, Any] = {
        "hash": preset_hash,
        "messages": request_messages,
    }

    headers = {
        "project": "WANTED_DATA",
        "apiKey": api_key,
        "Content-Type": "application/json; charset=utf-8",
    }

    timeout = _read_timeout_seconds()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{base_url}/api/preset/v2/chat/completions", json=payload, headers=headers
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"LAAS request failed: status={response.status_code}, body={response.text[:1000]}"
            ) from exc
        data = response.json()
    return _parse_laas_response_to_model_response(data)


def _build_openai_compatible_model() -> Model:
    base_url = os.getenv("LLM_OPENAI_BASE_URL", "").strip().rstrip("/")
    api_key = os.getenv("LLM_OPENAI_API_KEY", "").strip()
    model_name = os.getenv("LLM_OPENAI_MODEL", "glm-4.7").strip() or "glm-4.7"

    if not base_url:
        raise ValueError("LLM_OPENAI_BASE_URL is required when LLM_PROVIDER=openai_compatible")
    if not api_key:
        raise ValueError("LLM_OPENAI_API_KEY is required when LLM_PROVIDER=openai_compatible")

    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIChatModel(model_name=cast(Any, model_name), provider=provider)


def _build_laas_model() -> Model:
    return FunctionModel(function=_laas_function_model, model_name="laas-chat")


def build_model_for_env() -> Model:
    """Build a PydanticAI model from env config."""
    provider = os.getenv("LLM_PROVIDER", "laas").strip().lower()
    if provider == "openai_compatible":
        return _build_openai_compatible_model()
    if provider == "laas":
        return _build_laas_model()
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
