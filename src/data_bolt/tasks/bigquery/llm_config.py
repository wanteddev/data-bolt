"""LLM provider configuration and secret loading."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_ssm import SSMClient

LAAS_DEFAULT_BASE_URL = "https://api-laas.wanted.co.kr"
LAAS_EMPTY_PRESET_HASH = "2e1cfa82b035c26cbbbdae632cea070514eb8b773f616aaeaf668e2f0be8f10d"
LAAS_API_KEY_SSM_PARAM = "/DATA/PIPELINE/API_KEY/OPENAI"
LAAS_RAG_SCHEMA_COLLECTION = "RAG_DATA_CATALOG"
LAAS_RAG_GLOSSARY_COLLECTION = "RAG_GLOSSARY"

LLM_PROVIDER_DEFAULT = "laas"
LLM_PROVIDER_LAAS = "laas"
LLM_PROVIDER_OPENAI_COMPATIBLE = "openai_compatible"
LLM_PROVIDER_ANTHROPIC_COMPATIBLE = "anthropic_compatible"

OPENAI_COMPATIBLE_DEFAULT_MODEL = "glm-4.7"
ANTHROPIC_COMPATIBLE_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

LLM_TIMEOUT_SECONDS_DEFAULT = 60.0
LLM_TIMEOUT_INTENT_SECONDS_DEFAULT = 45.0
LLM_TIMEOUT_CHAT_PLANNER_SECONDS_DEFAULT = 45.0
LLM_TIMEOUT_REFINE_SECONDS_DEFAULT = 60.0


class _SSMParameterLoader:
    def __init__(self, cache_ttl: int = 300):
        self._cache: dict[str, tuple[float, str]] = {}
        self._cache_ttl = cache_ttl
        self._client: SSMClient | None = None

    def get_parameter(self, key: str, with_decryption: bool = False) -> str:
        cache_key = f"{key}:{with_decryption}"
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached:
            expires_at, value = cached
            if now < expires_at:
                return value

        if self._client is None:
            import boto3

            self._client = boto3.client("ssm")

        response = self._client.get_parameter(Name=key, WithDecryption=with_decryption)
        value = str(response["Parameter"]["Value"])
        self._cache[cache_key] = (now + self._cache_ttl, value)
        return value


_ssm_loader = _SSMParameterLoader()


def _get_laas_api_key() -> str:
    env_key = os.getenv("LAAS_API_KEY")
    if env_key:
        return env_key
    param_key = os.getenv("LAAS_API_KEY_SSM_PARAM", LAAS_API_KEY_SSM_PARAM)
    return _ssm_loader.get_parameter(param_key, True)


def _get_llm_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER_DEFAULT).strip().lower()
    if provider in {
        LLM_PROVIDER_LAAS,
        LLM_PROVIDER_OPENAI_COMPATIBLE,
        LLM_PROVIDER_ANTHROPIC_COMPATIBLE,
    }:
        return provider
    return LLM_PROVIDER_DEFAULT


def _get_openai_compatible_api_key() -> str:
    value = os.getenv("LLM_OPENAI_API_KEY", "").strip()
    if value:
        return value
    raise ValueError("LLM_OPENAI_API_KEY is required when LLM_PROVIDER=openai_compatible")


def _get_openai_compatible_base_url() -> str:
    value = os.getenv("LLM_OPENAI_BASE_URL", "").strip().rstrip("/")
    if value:
        return value
    raise ValueError("LLM_OPENAI_BASE_URL is required when LLM_PROVIDER=openai_compatible")


def _get_openai_compatible_model() -> str:
    model = os.getenv("LLM_OPENAI_MODEL", OPENAI_COMPATIBLE_DEFAULT_MODEL).strip()
    return model or OPENAI_COMPATIBLE_DEFAULT_MODEL


def _get_anthropic_compatible_api_key() -> str:
    value = os.getenv("LLM_ANTHROPIC_API_KEY", "").strip()
    if value:
        return value
    raise ValueError("LLM_ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic_compatible")


def _get_anthropic_compatible_base_url() -> str:
    value = os.getenv("LLM_ANTHROPIC_BASE_URL", "").strip().rstrip("/")
    if value:
        return value
    raise ValueError("LLM_ANTHROPIC_BASE_URL is required when LLM_PROVIDER=anthropic_compatible")


def _get_anthropic_compatible_model() -> str:
    model = os.getenv("LLM_ANTHROPIC_MODEL", ANTHROPIC_COMPATIBLE_DEFAULT_MODEL).strip()
    return model or ANTHROPIC_COMPATIBLE_DEFAULT_MODEL


def _read_timeout_env_optional(var_name: str) -> float | None:
    raw = os.getenv(var_name)
    if raw is None:
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _get_llm_timeout_seconds(use_case: str) -> float:
    common = _read_timeout_env_optional("LLM_TIMEOUT_SECONDS")
    if use_case == "intent":
        specific = _read_timeout_env_optional("LLM_TIMEOUT_INTENT_SECONDS")
        return specific if specific is not None else common or LLM_TIMEOUT_INTENT_SECONDS_DEFAULT
    if use_case == "planner":
        specific = _read_timeout_env_optional("LLM_TIMEOUT_CHAT_PLANNER_SECONDS")
        return (
            specific if specific is not None else common or LLM_TIMEOUT_CHAT_PLANNER_SECONDS_DEFAULT
        )
    if use_case == "refine":
        specific = _read_timeout_env_optional("LLM_TIMEOUT_REFINE_SECONDS")
        return specific if specific is not None else common or LLM_TIMEOUT_REFINE_SECONDS_DEFAULT
    specific = _read_timeout_env_optional("LLM_TIMEOUT_GENERATION_SECONDS")
    return specific if specific is not None else common or LLM_TIMEOUT_SECONDS_DEFAULT
