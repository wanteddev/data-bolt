"""HTTP transport for LLM providers."""

from __future__ import annotations

import os
from typing import Any

import httpx

from .llm_config import (
    LAAS_DEFAULT_BASE_URL,
    LAAS_EMPTY_PRESET_HASH,
    LLM_PROVIDER_ANTHROPIC_COMPATIBLE,
    LLM_PROVIDER_OPENAI_COMPATIBLE,
    _get_anthropic_compatible_api_key,
    _get_anthropic_compatible_base_url,
    _get_anthropic_compatible_model,
    _get_laas_api_key,
    _get_llm_provider,
    _get_openai_compatible_api_key,
    _get_openai_compatible_base_url,
    _get_openai_compatible_model,
)
from .parser import _extract_text_content
from .types import JsonValue


def _laas_post(path: str, payload: dict[str, Any], timeout: float) -> JsonValue:
    base_url = os.getenv("LAAS_BASE_URL", LAAS_DEFAULT_BASE_URL).rstrip("/")
    api_key = _get_laas_api_key()

    url = f"{base_url}{path}"
    headers = {
        "project": "WANTED_DATA",
        "apiKey": api_key,
        "Content-Type": "application/json; charset=utf-8",
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, (dict, list)):
            return data
        return {"content": data}


def _openai_compatible_post(path: str, payload: dict[str, Any], timeout: float) -> JsonValue:
    base_url = _get_openai_compatible_base_url()
    api_key = _get_openai_compatible_api_key()

    url = f"{base_url}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json; charset=utf-8",
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, (dict, list)):
            return data
        return {"content": data}


def _anthropic_compatible_post(path: str, payload: dict[str, Any], timeout: float) -> JsonValue:
    base_url = _get_anthropic_compatible_base_url()
    api_key = _get_anthropic_compatible_api_key()

    url = f"{base_url}{path}"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json; charset=utf-8",
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, (dict, list)):
            return data
        return {"content": data}


def _to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    out_messages: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = _extract_text_content(content)
            if text:
                system_parts.append(text)
            continue
        if role not in {"user", "assistant"}:
            continue
        text = _extract_text_content(content)
        if not text:
            continue
        out_messages.append({"role": role, "content": text})
    return ("\n\n".join(system_parts) if system_parts else None), out_messages


def _llm_chat_completion(
    *,
    messages: list[dict[str, Any]],
    timeout: float,
) -> JsonValue:
    provider = _get_llm_provider()
    if provider == LLM_PROVIDER_ANTHROPIC_COMPATIBLE:
        system, anthropic_messages = _to_anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": _get_anthropic_compatible_model(),
            "max_tokens": int(os.getenv("LLM_ANTHROPIC_MAX_TOKENS", "4096")),
            "messages": anthropic_messages or [{"role": "user", "content": ""}],
        }
        if system:
            payload["system"] = system
        return _anthropic_compatible_post("/v1/messages", payload, timeout=timeout)
    if provider == LLM_PROVIDER_OPENAI_COMPATIBLE:
        payload = {
            "model": _get_openai_compatible_model(),
            "messages": messages,
        }
        return _openai_compatible_post("/chat/completions", payload, timeout=timeout)

    payload = {"hash": LAAS_EMPTY_PRESET_HASH, "messages": messages}
    return _laas_post("/api/preset/v2/chat/completions", payload, timeout=timeout)
