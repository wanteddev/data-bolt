"""Model output parsing helpers for loose/legacy payload formats."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _first_non_empty_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return None


def _next_action_description(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, Mapping):
        explicit = _first_non_empty_string(
            value.get("description"),
            value.get("message"),
            value.get("summary"),
            value.get("next_step"),
            value.get("설명"),
            value.get("요약"),
        )
        if explicit:
            return explicit
        action_name = _first_non_empty_string(
            value.get("action"),
            value.get("name"),
            value.get("동작"),
        )
        sql = _first_non_empty_string(value.get("sql"), value.get("query"), value.get("쿼리"))
        if action_name and sql:
            compact_sql = " ".join(sql.split())
            if len(compact_sql) > 140:
                compact_sql = f"{compact_sql[:137]}..."
            return f"{action_name}: {compact_sql}"
        if action_name:
            action_map = {
                "bigquery_dry_run": "SQL 검증(dry-run)을 진행하겠습니다.",
                "bigquery_execute": "쿼리 실행 단계를 진행하겠습니다.",
                "get_schema_context": "관련 스키마를 먼저 조회하겠습니다.",
            }
            lowered = action_name.strip().lower()
            return action_map.get(lowered) or f"{action_name} 실행 예정"
    return None


def _summarize_table_candidates(value: Any) -> str | None:
    if not isinstance(value, list):
        return None

    table_names: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        schema_name = _first_non_empty_string(item.get("schema"), item.get("dataset"))
        table_name = _first_non_empty_string(
            item.get("name"),
            item.get("table"),
            item.get("table_name"),
            item.get("테이블명"),
        )
        column_name = _first_non_empty_string(
            item.get("column"), item.get("column_name"), item.get("컬럼")
        )
        if table_name is None:
            continue
        qualified = f"{schema_name}.{table_name}" if schema_name else table_name
        if column_name:
            qualified = f"{qualified}.{column_name}"
        if qualified not in table_names:
            table_names.append(qualified)
        if len(table_names) >= 6:
            break

    if not table_names:
        return None
    joined = ", ".join(table_names)
    return f"확인된 관련 테이블: {joined}"


def _summarize_string_list(value: Any, *, max_items: int = 3) -> str | None:
    if not isinstance(value, list):
        return None
    items = [str(item).strip() for item in value if isinstance(item, str) and item.strip()]
    if not items:
        return None
    selected = items[:max_items]
    if len(selected) == 1:
        return selected[0]
    return "\n".join(f"- {item}" for item in selected)


def _extract_wrapped_tool_sql(parsed: Mapping[str, Any]) -> tuple[str | None, str | None]:
    for tool_name in ("bigquery_dry_run", "bigquery_execute"):
        raw_tool_payload = parsed.get(tool_name)
        tool_payload = raw_tool_payload if isinstance(raw_tool_payload, Mapping) else {}
        sql_text = _first_non_empty_string(
            tool_payload.get("sql"),
            tool_payload.get("query"),
            tool_payload.get("statement"),
        )
        if sql_text:
            return sql_text, tool_name
    return None, None


def apply_parsed_output_dict(
    *,
    parsed: Mapping[str, Any],
    result: dict[str, Any],
    fallback_text: str = "",
) -> bool:
    nested_result_raw = parsed.get("result")
    nested_result = nested_result_raw if isinstance(nested_result_raw, Mapping) else {}

    kind = str(parsed.get("kind") or "").strip().lower()
    ask_message = _first_non_empty_string(parsed.get("message"), parsed.get("text"))
    if kind == "ask_user" and ask_message:
        result["action"] = "ask_user"
        result["response_text"] = ask_message
        return True

    if kind == "reply":
        response_message = _first_non_empty_string(
            parsed.get("message"),
            parsed.get("reply"),
            parsed.get("answer"),
            parsed.get("답변"),
            parsed.get("설명"),
            parsed.get("요약"),
            parsed.get("comment"),
            parsed.get("error"),
            parsed.get("response"),
            parsed.get("text"),
            parsed.get("explanation"),
            parsed.get("thought"),
            parsed.get("thoughts"),
            nested_result.get("message"),
            nested_result.get("reply"),
            nested_result.get("answer"),
            nested_result.get("답변"),
            nested_result.get("설명"),
            nested_result.get("요약"),
            fallback_text,
        )
        if response_message:
            result["action"] = "reply"
            result["response_text"] = response_message
            sql_raw = parsed.get("sql")
            if isinstance(sql_raw, str) and sql_raw.strip():
                result["candidate_sql"] = sql_raw
            result["generation_result"] = {
                "answer_structured": {
                    "sql": result.get("candidate_sql"),
                    "explanation": response_message,
                }
            }
            return True

    final_ask_raw = parsed.get("final_result_AskUser")
    final_ask = final_ask_raw if isinstance(final_ask_raw, Mapping) else {}
    ask_message = _first_non_empty_string(
        final_ask.get("message"),
        final_ask.get("text"),
        final_ask.get("response"),
        final_ask.get("reply"),
        final_ask.get("answer"),
        final_ask.get("설명"),
    )
    if ask_message:
        result["action"] = "ask_user"
        result["response_text"] = ask_message
        return True

    final_reply_raw = parsed.get("final_result_AnalystReply")
    final_reply = final_reply_raw if isinstance(final_reply_raw, Mapping) else {}
    if final_reply:
        response_message = _first_non_empty_string(
            final_reply.get("message"),
            final_reply.get("reply"),
            final_reply.get("answer"),
            final_reply.get("답변"),
            final_reply.get("설명"),
            final_reply.get("요약"),
            final_reply.get("comment"),
            final_reply.get("error"),
            final_reply.get("response"),
            final_reply.get("text"),
            final_reply.get("explanation"),
            final_reply.get("thought"),
            final_reply.get("thoughts"),
            nested_result.get("message"),
            nested_result.get("reply"),
            nested_result.get("answer"),
            nested_result.get("답변"),
            nested_result.get("설명"),
            nested_result.get("요약"),
            fallback_text,
        )
        if response_message:
            result["action"] = "reply"
            result["response_text"] = response_message
            sql_raw = final_reply.get("sql")
            if isinstance(sql_raw, str) and sql_raw.strip():
                result["candidate_sql"] = sql_raw
            result["generation_result"] = {
                "answer_structured": {
                    "sql": result.get("candidate_sql"),
                    "explanation": response_message,
                }
            }
            return True

    response_message = _first_non_empty_string(
        parsed.get("message"),
        parsed.get("reply"),
        parsed.get("answer"),
        parsed.get("답변"),
        parsed.get("설명"),
        parsed.get("요약"),
        parsed.get("comment"),
        parsed.get("error"),
        parsed.get("response"),
        parsed.get("text"),
        parsed.get("explanation"),
        parsed.get("thought"),
        parsed.get("thoughts"),
        nested_result.get("message"),
        nested_result.get("reply"),
        nested_result.get("answer"),
        nested_result.get("답변"),
        nested_result.get("설명"),
        nested_result.get("요약"),
    )
    list_summary = _summarize_string_list(parsed.get("criteria")) or _summarize_string_list(
        parsed.get("plan")
    )
    if response_message is None and list_summary is not None:
        response_message = list_summary

    table_summary = _summarize_table_candidates(
        parsed.get("tables")
    ) or _summarize_table_candidates(parsed.get("추천_테이블"))
    if table_summary is None:
        table_summary = _summarize_table_candidates(
            nested_result.get("tables")
        ) or _summarize_table_candidates(nested_result.get("추천_테이블"))
    if response_message is None and table_summary is not None:
        response_message = table_summary

    next_action_description = _next_action_description(
        parsed.get("next_action")
    ) or _next_action_description(nested_result.get("next_action"))
    if response_message is None and next_action_description:
        response_message = next_action_description

    sql_raw = _first_non_empty_string(
        parsed.get("sql"), parsed.get("query"), parsed.get("statement")
    )
    wrapped_sql, wrapped_tool_name = _extract_wrapped_tool_sql(parsed)
    if wrapped_sql is None:
        wrapped_sql, wrapped_tool_name = _extract_wrapped_tool_sql(nested_result)
    nested_sql = _first_non_empty_string(
        nested_result.get("sql"),
        nested_result.get("query"),
        nested_result.get("statement"),
    )
    resolved_sql = sql_raw or wrapped_sql
    if resolved_sql is None:
        resolved_sql = nested_sql
    if response_message or resolved_sql:
        used_default_message = response_message is None
        resolved_message = response_message or "요청하신 SQL을 준비했습니다."
        if (
            next_action_description
            and next_action_description not in resolved_message
            and not resolved_sql
        ):
            resolved_message = f"{resolved_message}\n다음 단계: {next_action_description}"
        if wrapped_tool_name and next_action_description is None:
            resolved_message = (
                f"{resolved_message}\n다음 단계: {wrapped_tool_name}로 검증할 수 있습니다."
            )
        if resolved_sql and used_default_message and "```sql" not in resolved_message:
            resolved_message = f"{resolved_message}\n```sql\n{resolved_sql}\n```"
        result["action"] = "reply"
        result["response_text"] = resolved_message
        if resolved_sql:
            result["candidate_sql"] = resolved_sql
        result["generation_result"] = {
            "answer_structured": {
                "sql": result.get("candidate_sql"),
                "explanation": resolved_message,
            }
        }
        return True

    return False
