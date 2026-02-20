"""Result contract mapping from agent outputs."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from pydantic_ai import DeferredToolRequests

from ..approval_store import ApprovalContext, build_approval_context_timestamps
from ..deps import AnalystDeps
from ..models import AnalystReply, AskUser, DryRunResult, QueryResultSummary
from .output_parsing import apply_parsed_output_dict, parse_json_object

type SaveApprovalContext = Callable[[ApprovalContext], None]


def dry_run_dict(dry_run: DryRunResult | None) -> dict[str, Any]:
    if dry_run is None:
        return {}
    return dry_run.model_dump(mode="json")


def execution_dict(result: QueryResultSummary | None) -> dict[str, Any]:
    if result is None:
        return {}
    return result.model_dump(mode="json")


def base_result() -> dict[str, Any]:
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


def approval_response_text(deferred: DeferredToolRequests) -> str:
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


def approval_blocks(approval_request_id: str, message: str) -> list[dict[str, Any]]:
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


def _build_approval_context(
    *,
    approval_request_id: str,
    deferred: DeferredToolRequests,
    run_messages_json: str,
    payload: Mapping[str, Any],
) -> ApprovalContext:
    created_at, expires_at = build_approval_context_timestamps()
    tool_call_ids = [call.tool_call_id for call in deferred.approvals]
    return ApprovalContext(
        approval_request_id=approval_request_id,
        requester_user_id=str(payload.get("user_id") or ""),
        channel_id=str(payload.get("channel_id") or ""),
        thread_ts=str(payload.get("thread_ts") or ""),
        team_id=str(payload.get("team_id") or ""),
        tool_call_ids=tool_call_ids,
        deferred_metadata=deferred.metadata,
        session_messages_json=run_messages_json,
        created_at=created_at,
        expires_at=expires_at,
    )


def apply_output_to_result(
    *,
    output: Any,
    deps: AnalystDeps,
    run_messages_json: str,
    payload: Mapping[str, Any],
    save_approval_context: SaveApprovalContext | None = None,
) -> dict[str, Any]:
    result = base_result()
    result["candidate_sql"] = deps.last_sql
    result["validation"] = dry_run_dict(deps.last_dry_run)
    result["execution"] = execution_dict(deps.last_result)

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
        context = _build_approval_context(
            approval_request_id=approval_request_id,
            deferred=output,
            run_messages_json=run_messages_json,
            payload=payload,
        )
        if save_approval_context is not None:
            save_approval_context(context)

        approval_text = approval_response_text(output)
        result["action"] = "approval_required"
        result["response_text"] = approval_text
        result["requires_approval"] = True
        result["approval_request_id"] = approval_request_id
        result["response_blocks"] = approval_blocks(approval_request_id, approval_text)
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

    if isinstance(output, Mapping):
        if apply_parsed_output_dict(parsed=output, result=result):
            return result
        result["action"] = "reply"
        result["response_text"] = json.dumps(output, ensure_ascii=False, default=str)
        result["generation_result"] = {
            "answer_structured": {
                "sql": result.get("candidate_sql"),
                "explanation": result["response_text"],
            }
        }
        return result

    if isinstance(output, str):
        parsed = parse_json_object(output)
        if apply_parsed_output_dict(parsed=parsed, result=result, fallback_text=output):
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
