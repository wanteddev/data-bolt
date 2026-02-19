"""Public service entrypoints for the PydanticAI-based analyst agent."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from typing import Any, cast

from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    UsageLimits,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
)

from data_bolt.tasks.tools import lookup_schema_rag_context

from .approval_store import ApprovalContext, DynamoApprovalStore, build_approval_context_timestamps
from .deps import AnalystDeps, QueryPolicy, SchemaRetriever
from .model_factory import build_model_for_env
from .models import (
    AnalystReply,
    AskUser,
    DryRunResult,
    QueryResultSummary,
    SchemaContext,
    TableSnippet,
)
from .tools import tool_bigquery_dry_run, tool_bigquery_execute, tool_get_schema_context

OUTPUT_TYPES = [AskUser, AnalystReply, str, DeferredToolRequests]

DEFAULT_USAGE_LIMITS = UsageLimits(
    tool_calls_limit=12,
    request_limit=6,
)

_agent: Agent[AnalystDeps, Any] | None = None


class _DefaultSchemaRetriever(SchemaRetriever):
    def search(self, question: str, top_k: int = 6) -> SchemaContext:
        del top_k
        raw = lookup_schema_rag_context(question)
        table_info = str(raw.get("table_info") or "")
        glossary_info = str(raw.get("glossary_info") or "")
        raw_meta = raw.get("meta")
        meta = raw_meta if isinstance(raw_meta, dict) else {}

        snippets: list[TableSnippet] = []
        if table_info.strip():
            snippets.append(
                TableSnippet(
                    table="rag_context",
                    description=table_info.strip()[:4000],
                )
            )
        notes = [glossary_info.strip()] if glossary_info.strip() else []
        return SchemaContext(
            snippets=snippets,
            notes=notes,
            raw_table_info=table_info,
            raw_glossary_info=glossary_info,
            meta=meta,
        )


def _build_agent() -> Agent[AnalystDeps, Any]:
    agent = Agent(
        build_model_for_env(),
        deps_type=AnalystDeps,
        output_type=OUTPUT_TYPES,
        instructions=(
            "You are a senior data analyst speaking natural Korean.\n"
            "Use BigQuery SQL to answer business data questions.\n"
            "Rules:\n"
            "- If schema context is needed, call get_schema_context first.\n"
            "- Before execute, always validate SQL with bigquery_dry_run.\n"
            "- If information is missing, ask concise clarification questions.\n"
            "- Keep SQL read-only unless user explicitly asks for writes.\n"
            "- For expensive or risky execution, explain approval requirement clearly.\n"
            "- Always return valid JSON output that matches the requested schema.\n"
            "- Return concise Korean responses with practical next steps.\n"
        ),
        output_retries=2,
        defer_model_check=True,
    )
    agent_any = cast(Any, agent)
    agent_any.tool(tool_get_schema_context, name="get_schema_context")
    agent_any.tool(tool_bigquery_dry_run, name="bigquery_dry_run")
    agent_any.tool(tool_bigquery_execute, name="bigquery_execute")
    return agent


def _get_agent() -> Agent[AnalystDeps, Any]:
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _build_policy() -> QueryPolicy:
    return QueryPolicy(
        max_bytes_without_approval=_env_int(
            "BIGQUERY_AGENT_MAX_BYTES_WITHOUT_APPROVAL", 5 * 1024**3
        ),
        max_bytes_hard_limit=_env_int("BIGQUERY_AGENT_MAX_BYTES_HARD_LIMIT", 50 * 1024**3),
        preview_rows=_env_int("BIGQUERY_RESULT_PREVIEW_ROWS", 50),
        require_approval_for_dml_ddl=True,
    )


def _build_deps(payload: Mapping[str, Any]) -> AnalystDeps:
    return AnalystDeps(
        bq_client=None,
        schema_retriever=_DefaultSchemaRetriever(),
        policy=_build_policy(),
        default_project=os.getenv("BIGQUERY_PROJECT_ID") or None,
        default_dataset=os.getenv("BIGQUERY_DATASET") or None,
        location=os.getenv("BIGQUERY_LOCATION") or None,
        requester_user_id=str(payload.get("user_id") or ""),
        channel_id=str(payload.get("channel_id") or ""),
        thread_ts=str(payload.get("thread_ts") or ""),
    )


def _history_to_model_messages(history: Any) -> list[ModelMessage]:
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


def _dry_run_dict(dry_run: DryRunResult | None) -> dict[str, Any]:
    if dry_run is None:
        return {}
    return dry_run.model_dump(mode="json")


def _execution_dict(result: QueryResultSummary | None) -> dict[str, Any]:
    if result is None:
        return {}
    return result.model_dump(mode="json")


def _base_result() -> dict[str, Any]:
    return {
        "action": "reply",
        "should_respond": True,
        "response_text": "",
        "candidate_sql": None,
        "validation": {},
        "execution": {},
        "generation_result": {},
        "error": None,
    }


def _recover_from_tool_retry_error(exc: Exception, deps: AnalystDeps) -> dict[str, Any] | None:
    message = str(exc)
    if "exceeded max retries count" not in message.lower():
        return None

    result = _base_result()
    result["candidate_sql"] = deps.last_sql
    result["validation"] = _dry_run_dict(deps.last_dry_run)
    result["execution"] = _execution_dict(deps.last_result)

    dry_run_error = deps.last_dry_run.error if deps.last_dry_run else None
    if dry_run_error:
        result["action"] = "reply"
        result["response_text"] = (
            "쿼리 검증 단계에서 오류가 발생했습니다.\n"
            f"{dry_run_error}\n"
            "권한/인증 상태를 확인한 뒤 다시 시도해 주세요."
        )
    else:
        result["action"] = "ask_user"
        result["response_text"] = (
            "요청을 처리하는 중 도구 호출 형식이 맞지 않았습니다. "
            "기간/지표/테이블명을 조금 더 구체적으로 알려주세요."
        )

    result["generation_result"] = {"meta": {"recovered_from_tool_retry_error": True}}
    result["error"] = None
    return result


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


def _approval_response_text(deferred: DeferredToolRequests) -> str:
    if not deferred.approvals:
        return "도구 실행 승인이 필요합니다."

    approval = deferred.approvals[0]
    meta = deferred.metadata.get(approval.tool_call_id, {}) if deferred.metadata else {}
    reason = str(meta.get("reason") or "policy")

    if reason == "cost":
        estimated_bytes = meta.get("estimated_bytes")
        threshold = meta.get("threshold")
        return (
            "예상 비용이 승인 임계값을 초과했습니다.\n"
            f"estimated_bytes={estimated_bytes}, threshold={threshold}\n"
            "`실행 승인` 버튼 또는 `실행 취소` 버튼을 선택해 주세요."
        )
    if reason == "dml_ddl_or_non_select":
        statement_type = meta.get("statement_type")
        return (
            "쓰기/DDL 또는 non-SELECT 쿼리는 승인 후에만 실행됩니다.\n"
            f"statement_type={statement_type}\n"
            "`실행 승인` 버튼 또는 `실행 취소` 버튼을 선택해 주세요."
        )
    return "쿼리 실행을 위해 승인이 필요합니다. `실행 승인` 또는 `실행 취소`를 선택해 주세요."


def _approval_blocks(approval_request_id: str, message: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": message,
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "style": "primary",
                    "text": {"type": "plain_text", "text": "실행 승인"},
                    "action_id": "bq_approve_execute",
                    "value": approval_request_id,
                },
                {
                    "type": "button",
                    "style": "danger",
                    "text": {"type": "plain_text", "text": "실행 취소"},
                    "action_id": "bq_deny_execute",
                    "value": approval_request_id,
                },
            ],
        },
    ]


def _apply_output_to_result(
    *,
    output: Any,
    deps: AnalystDeps,
    run_messages_json: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    result = _base_result()
    result["candidate_sql"] = deps.last_sql
    result["validation"] = _dry_run_dict(deps.last_dry_run)
    result["execution"] = _execution_dict(deps.last_result)

    if isinstance(output, AskUser):
        result["action"] = "ask_user"
        result["response_text"] = output.message
        return result

    if isinstance(output, AnalystReply):
        result["action"] = "reply"
        result["response_text"] = output.message
        if output.sql:
            result["candidate_sql"] = output.sql
        if output.dry_run is not None:
            result["validation"] = output.dry_run.model_dump(mode="json")
        if output.result is not None:
            result["execution"] = output.result.model_dump(mode="json")
        result["generation_result"] = {
            "answer_structured": {
                "sql": result.get("candidate_sql"),
                "explanation": output.message,
                "next_suggestions": output.next_suggestions,
            }
        }
        return result

    if isinstance(output, DeferredToolRequests):
        approval_request_id = str(uuid.uuid4())
        created_at, expires_at = build_approval_context_timestamps()
        store = DynamoApprovalStore.from_env()

        tool_call_ids = [call.tool_call_id for call in output.approvals]
        context = ApprovalContext(
            approval_request_id=approval_request_id,
            requester_user_id=str(payload.get("user_id") or ""),
            channel_id=str(payload.get("channel_id") or ""),
            thread_ts=str(payload.get("thread_ts") or ""),
            team_id=str(payload.get("team_id") or ""),
            tool_call_ids=tool_call_ids,
            deferred_metadata=output.metadata,
            session_messages_json=run_messages_json,
            created_at=created_at,
            expires_at=expires_at,
        )
        store.save(context)

        approval_text = _approval_response_text(output)
        result["action"] = "approval_required"
        result["response_text"] = approval_text
        result["requires_approval"] = True
        result["approval_request_id"] = approval_request_id
        result["response_blocks"] = _approval_blocks(approval_request_id, approval_text)
        result["approval_requests"] = [
            {
                "tool_name": call.tool_name,
                "tool_call_id": call.tool_call_id,
                "args": call.args_as_dict(),
                "metadata": output.metadata.get(call.tool_call_id, {}),
            }
            for call in output.approvals
        ]
        result["generation_result"] = {
            "meta": {
                "approval_required": True,
                "approval_request_id": approval_request_id,
            }
        }
        return result

    if isinstance(output, str):
        parsed = _parse_json_object(output)
        final_ask_raw = parsed.get("final_result_AskUser")
        final_ask = final_ask_raw if isinstance(final_ask_raw, dict) else {}
        ask_message = final_ask.get("message")
        if not isinstance(ask_message, str):
            ask_message = final_ask.get("text")
        if isinstance(ask_message, str) and ask_message.strip():
            result["action"] = "ask_user"
            result["response_text"] = ask_message
            return result

        final_reply_raw = parsed.get("final_result_AnalystReply")
        final_reply = final_reply_raw if isinstance(final_reply_raw, dict) else {}
        if final_reply:
            message = final_reply.get("message")
            explanation = final_reply.get("explanation")
            response_message = (
                message
                if isinstance(message, str) and message.strip()
                else explanation
                if isinstance(explanation, str) and explanation.strip()
                else output
            )
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
            return result

        response_message = parsed.get("message")
        explanation = parsed.get("explanation")
        sql = parsed.get("sql")
        if (
            isinstance(response_message, str)
            or isinstance(explanation, str)
            or isinstance(sql, str)
        ):
            resolved_message = (
                response_message
                if isinstance(response_message, str) and response_message.strip()
                else explanation
                if isinstance(explanation, str) and explanation.strip()
                else output
            )
            result["action"] = "reply"
            result["response_text"] = resolved_message
            if isinstance(sql, str) and sql.strip():
                result["candidate_sql"] = sql
            result["generation_result"] = {
                "answer_structured": {
                    "sql": result.get("candidate_sql"),
                    "explanation": resolved_message,
                }
            }
            return result

    result["action"] = "reply"
    result["response_text"] = str(output)
    result["generation_result"] = {
        "answer_structured": {
            "sql": result.get("candidate_sql"),
            "explanation": str(output),
        }
    }
    return result


def run_analyst_turn(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one analyst turn and return Slack-friendly result contract."""
    text = str(payload.get("text") or payload.get("question") or "").strip()
    if not text:
        return {
            **_base_result(),
            "should_respond": False,
            "response_text": "",
            "action": "ignore",
        }

    deps = _build_deps(payload)
    history = _history_to_model_messages(payload.get("history"))

    try:
        run_result = _get_agent().run_sync(
            user_prompt=text,
            deps=deps,
            message_history=history,
            usage_limits=DEFAULT_USAGE_LIMITS,
        )
        run_messages_json = run_result.all_messages_json().decode("utf-8")
        return _apply_output_to_result(
            output=run_result.output,
            deps=deps,
            run_messages_json=run_messages_json,
            payload=payload,
        )
    except Exception as exc:
        recovered = _recover_from_tool_retry_error(exc, deps)
        if recovered is not None:
            return recovered
        return {
            **_base_result(),
            "action": "error",
            "error": str(exc),
            "response_text": f":x: 요청 처리 중 오류가 발생했습니다.\n{exc}",
        }


def run_analyst_approval(payload: dict[str, Any]) -> dict[str, Any]:
    """Resume a deferred approval request and continue execution."""
    approval_request_id = str(payload.get("approval_request_id") or "").strip()
    approved = bool(payload.get("approved"))
    actor_user_id = str(payload.get("user_id") or "")

    if not approval_request_id:
        return {
            **_base_result(),
            "action": "error",
            "error": "approval_request_id is required",
            "response_text": ":x: 승인 요청 식별자가 없습니다.",
        }

    deps: AnalystDeps | None = None
    try:
        store = DynamoApprovalStore.from_env()
        context = store.load(approval_request_id)
        if context is None:
            return {
                **_base_result(),
                "action": "error",
                "error": "approval request not found",
                "response_text": ":warning: 승인 대기 요청을 찾을 수 없습니다. 다시 요청해주세요.",
            }

        if (
            actor_user_id
            and context.requester_user_id
            and actor_user_id != context.requester_user_id
        ):
            return {
                **_base_result(),
                "action": "error",
                "error": "permission denied",
                "response_text": ":x: 원 요청자만 승인/취소할 수 있습니다.",
                "should_respond": True,
            }

        approvals = {tool_call_id: approved for tool_call_id in context.tool_call_ids}
        deferred_results = DeferredToolResults(
            approvals=cast(Any, approvals),
            metadata=context.deferred_metadata,
        )
        message_history = ModelMessagesTypeAdapter.validate_json(
            context.session_messages_json.encode("utf-8")
        )

        run_payload = {
            "user_id": context.requester_user_id,
            "channel_id": context.channel_id,
            "thread_ts": context.thread_ts,
            "team_id": context.team_id,
        }
        deps = _build_deps(run_payload)

        run_result = _get_agent().run_sync(
            deps=deps,
            message_history=message_history,
            deferred_tool_results=deferred_results,
            usage_limits=DEFAULT_USAGE_LIMITS,
        )

        store.delete(approval_request_id)
        run_messages_json = run_result.all_messages_json().decode("utf-8")
        result = _apply_output_to_result(
            output=run_result.output,
            deps=deps,
            run_messages_json=run_messages_json,
            payload=run_payload,
        )
        result["approval_request_id"] = approval_request_id
        result["approved"] = approved
        return result

    except Exception as exc:
        if deps is not None:
            recovered = _recover_from_tool_retry_error(exc, deps)
            if recovered is not None:
                return recovered
        return {
            **_base_result(),
            "action": "error",
            "error": str(exc),
            "response_text": f":x: 승인 처리 중 오류가 발생했습니다.\n{exc}",
        }
