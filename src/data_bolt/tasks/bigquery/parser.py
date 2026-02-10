"""Parsing helpers for LLM and SQL outputs."""

from __future__ import annotations

import json
import re
from typing import Any

from .types import JsonValue


def extract_sql_blocks(text: str, min_length: int = 20) -> list[str]:
    """Extract SQL blocks from a text blob."""
    if not text:
        return []

    blocks: list[str] = []

    fenced_pattern = re.compile(r"```(?:sql)?\n(.*?)```", re.S | re.I)
    for match in fenced_pattern.finditer(text):
        block = match.group(1).strip()
        if len(block) >= min_length:
            blocks.extend([b.strip() for b in block.split(";") if len(b.strip()) >= min_length])

    if blocks:
        return blocks

    sql_start_pattern = re.compile(r"^(SELECT|WITH|CREATE|INSERT|UPDATE|DELETE)\b", re.I)
    parts = [p.strip() for p in re.split(r";", text) if p.strip()]
    for part in parts:
        first_line = part.strip().splitlines()[0].strip() if part.strip() else ""
        if sql_start_pattern.search(first_line):
            blocks.append(part)

    return blocks


def _looks_like_sql(text: str) -> bool:
    if not text:
        return False
    if "```" in text:
        return True
    pat = re.compile(r"\b(SELECT|WITH|CREATE|INSERT|UPDATE|DELETE)\b", re.I)
    return bool(pat.search(text))


def _extract_text_content(content: Any) -> str | None:
    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _extract_text_content(item)
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)
        return None

    if isinstance(content, dict):
        for key in ("text", "content", "value", "output_text"):
            text = _extract_text_content(content.get(key))
            if text:
                return text
    return None


def _collect_candidate_contents(resp: JsonValue) -> list[str]:
    if not isinstance(resp, dict):
        return []

    candidates: list[str] = []

    for key in ("content", "text", "output_text"):
        text = _extract_text_content(resp.get(key))
        if text:
            candidates.append(text)

    message_text = _extract_text_content(resp.get("message"))
    if message_text:
        candidates.append(message_text)

    choices = resp.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                text = _extract_text_content(message.get("content"))
                if text:
                    candidates.append(text)
            text = _extract_text_content(choice.get("content"))
            if text:
                candidates.append(text)

    output = resp.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            text = _extract_text_content(item.get("content"))
            if text:
                candidates.append(text)

    return candidates


def _parse_json_response(content: str) -> dict[str, Any]:
    candidates = [content]
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.S | re.I)
    if fenced_match:
        candidates.append(fenced_match.group(1))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return {}


def _extract_sql_from_response(resp: JsonValue) -> tuple[str | None, str | None]:
    sql = resp.get("sql") if isinstance(resp, dict) else None
    explanation = resp.get("explanation") if isinstance(resp, dict) else None
    if isinstance(sql, str) and sql.strip():
        return sql.strip(), explanation

    contents = _collect_candidate_contents(resp)
    for content in contents:
        parsed = _parse_json_response(content)
        if parsed and "sql" in parsed:
            parsed_sql = parsed.get("sql")
            parsed_explanation = parsed.get("explanation")
            resolved_explanation = (
                parsed_explanation
                if isinstance(parsed_explanation, str)
                else explanation
                if isinstance(explanation, str)
                else None
            )
            if isinstance(parsed_sql, str):
                normalized_sql = parsed_sql.strip()
                return (normalized_sql or None), resolved_explanation
            return None, resolved_explanation

        if "```" in content:
            blocks = extract_sql_blocks(content)
            if blocks:
                return blocks[0], explanation

    return None, explanation
