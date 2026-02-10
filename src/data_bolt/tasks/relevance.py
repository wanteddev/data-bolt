"""Shared relevance heuristics for Slack message routing."""

from __future__ import annotations

import os
import re

CHANNEL_KEYWORDS_DEFAULT = (
    "bigquery,bq,sql,쿼리,빅쿼리,테이블,스키마,데이터,db,database,column,dry-run,dryrun"
)


def env_truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_csv_set(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def looks_like_sql(text: str) -> bool:
    if "```" in text:
        return True
    return bool(re.search(r"\b(SELECT|WITH|CREATE|INSERT|UPDATE|DELETE)\b", text, re.I))


def looks_like_data_request(text: str) -> bool:
    lowered = text.lower()
    keywords = parse_csv_set(
        os.getenv("SLACK_CHANNEL_AUTO_REPLY_KEYWORDS", CHANNEL_KEYWORDS_DEFAULT)
    )
    if any(keyword in lowered for keyword in keywords):
        return True
    if looks_like_sql(text):
        return True
    if "?" in text and any(token in lowered for token in ("sql", "query", "쿼리", "데이터")):
        return True
    return False


def channel_allowed(channel_id: str | None) -> bool:
    if not channel_id:
        return False
    allowlist = parse_csv_set(os.getenv("SLACK_CHANNEL_AUTO_REPLY_ALLOWLIST", ""))
    if not allowlist:
        return True
    return channel_id.lower() in allowlist


def should_respond_to_message(
    *,
    text: str,
    channel_type: str,
    is_mention: bool,
    is_thread_followup: bool,
    channel_id: str | None,
) -> bool:
    if is_mention or channel_type == "im":
        return True
    if channel_type not in {"channel", "group", "mpim"}:
        return False
    if not env_truthy(os.getenv("SLACK_CHANNEL_AUTO_REPLY", "true"), True):
        return False
    if not channel_allowed(channel_id):
        return False
    if is_thread_followup and env_truthy(
        os.getenv("SLACK_CHANNEL_AUTO_REPLY_IN_THREADS", "true"), True
    ):
        return True
    return looks_like_data_request(text)
