"""Recovery helpers for known runtime failures."""

from __future__ import annotations

import re
from typing import Any

from ..deps import AnalystDeps
from .result_contract import base_result, dry_run_dict, execution_dict


def recover_from_tool_retry_error(exc: Exception, deps: AnalystDeps) -> dict[str, Any] | None:
    message = str(exc)
    if "exceeded max retries count" not in message.lower():
        return None

    result = base_result()
    result["candidate_sql"] = deps.last_sql
    result["validation"] = dry_run_dict(deps.last_dry_run)
    result["execution"] = execution_dict(deps.last_result)

    dry_run_error = deps.last_dry_run.error if deps.last_dry_run else None
    if dry_run_error:
        result["action"] = "reply"
        result["response_text"] = (
            "쿼리 검증 단계에서 오류가 발생했습니다.\n"
            f"{dry_run_error}\n"
            "권한/인증 상태를 확인한 뒤 다시 시도해 주세요."
        )
    else:
        normalized_error = " ".join(message.split())
        tool_match = re.search(r"tool\s+'([^']+)'", message, flags=re.IGNORECASE)
        tool_name = tool_match.group(1) if tool_match else ""
        error_detail = normalized_error
        lower_error = normalized_error.lower()
        for marker in ("last error:", "last validation error:", "caused by:"):
            marker_index = lower_error.find(marker)
            if marker_index >= 0:
                error_detail = normalized_error[marker_index + len(marker) :].strip()
                break
        if len(error_detail) > 280:
            error_detail = f"{error_detail[:277]}..."

        result["action"] = "ask_user"
        if tool_name == "get_schema_context":
            result["response_text"] = (
                "스키마 검색 호출에서 필요한 입력이 부족하거나 형식이 맞지 않았습니다.\n"
                f"오류: {error_detail}\n"
                "분석할 지표와 기간을 한 문장으로 다시 알려주세요."
            )
        else:
            result["response_text"] = (
                "요청 처리 중 도구 호출에서 오류가 발생했습니다.\n"
                f"오류: {error_detail}\n"
                "원하는 지표/기간/테이블 후보를 조금 더 구체적으로 알려주세요."
            )

    result["generation_result"] = {
        "meta": {
            "recovered_from_tool_retry_error": True,
            "tool_retry_error": message,
        }
    }
    result["error"] = None
    return result


def recover_from_request_limit_error(exc: Exception, deps: AnalystDeps) -> dict[str, Any] | None:
    message = str(exc)
    lowered = message.lower()
    if "request_limit" not in lowered or "exceed" not in lowered:
        return None

    result = base_result()
    result["candidate_sql"] = deps.last_sql
    result["validation"] = dry_run_dict(deps.last_dry_run)
    result["execution"] = execution_dict(deps.last_result)
    result["action"] = "ask_user"
    result["response_text"] = (
        "요청 처리 중 내부 재시도 한도를 초과했습니다. "
        "질문 범위를 조금 좁혀서 다시 요청해 주세요. "
        "예: 기간/지표/국가 중 1~2개만 지정"
    )
    result["generation_result"] = {
        "meta": {
            "recovered_from_request_limit_error": True,
            "request_limit_error": message,
        }
    }
    result["error"] = None
    return result
