"""Prompt and message construction helpers."""

from __future__ import annotations

import os
from typing import Any

from .parser import _looks_like_sql


def _build_bigquery_sql_rules_block() -> str:
    return (
        """
[응답 규칙]
1) BigQuery 표준 SQL만 사용 (legacy 금지), SELECT * 금지, 스키마에 없는 컬럼 추측 금지.
2) 실행 가능한 단일 read-only statement(SELECT/WITH)만 생성하고 다중 statement 금지.
3) 한국 시간(Asia/Seoul) 고려 시 TIMESTAMP/DATETIME 변환을 명시.
   - `AT TIME ZONE` 문법은 사용하지 말고, BigQuery 함수(`DATE(ts, 'Asia/Seoul')`, `DATETIME(ts, 'Asia/Seoul')`)를 사용.
   - "지난 주(월~일)" 범위는 `DATE_TRUNC(CURRENT_DATE('Asia/Seoul'), WEEK(MONDAY))`를 기준으로 계산.
   - `TIMESTAMP_SUB(..., INTERVAL n WEEK)` 패턴 금지.
   - `DATE_TRUNC` 사용 시 DATE 타입에는 인자 2개만 허용됨: `DATE_TRUNC(date_expr, WEEK(MONDAY))`.
   - 금지 예시: `DATE_TRUNC(CURRENT_DATE('Asia/Seoul'), WEEK, 1)`.
   - 권장 예시:
     `DATE_SUB(DATE_TRUNC(CURRENT_DATE('Asia/Seoul'), WEEK(MONDAY)), INTERVAL 1 WEEK)` ~
     `DATE_TRUNC(CURRENT_DATE('Asia/Seoul'), WEEK(MONDAY))`.
4) 조인 시 키와 null 처리 근거를 설명에 적시.
5) 출력 JSON 스키마:
{
  "sql": "<BigQuery SQL>",
  "explanation": "<요청 해석, 조인/필터 근거, 시간대 처리 근거, 실행/비용 리스크>",
  "assumptions": "<제공되지 않은 가정 목록 또는 빈 배열/문자열>",
  "validation_steps": [
    "스키마 존재 확인 방법",
    "작은 기간으로 샘플 실행해 행수/NULL 비율 검증",
    "엣지 케이스 점검 아이디어"
  ]
}
6) COUNT(_) 금지, COUNT(*) 사용.
7) 불확실한 부분은 보수적으로 가정하고 explanation/assumptions에 명시.
8) dry-run 성공 가능성을 최우선으로 SQL 문법/구조를 구성.
9) 사용자가 명시적으로 실행을 요청하지 않았다면 실행 자체를 가정하지 말고 분석/검증 중심으로 설명한다.
        """
    ).strip()


def _build_bigquery_sql_process_block() -> str:
    return (
        """
[절차]
1) 요청 재진술
2) 컨텍스트 선택
3) 실행 가능 설계(조인/필터/비용)
4) SQL 생성
5) 자체 검증
        """
    ).strip()


def _build_refine_instruction_block() -> str:
    return (
        """
[Directive]
너는 **BigQuery 표준 SQL** 수정 전문 도우미다. dry-run 에러를 반영해 SQL을 고쳐라.
        """
        + "\n\n"
        + _build_bigquery_sql_rules_block()
        + "\n\n"
        + _build_bigquery_sql_process_block()
    ).strip()


def _classify_instruction_type(question: str, history: list[dict[str, Any]] | None) -> str:
    q = (question or "").lower()
    wants_analysis = any(
        k in q
        for k in [
            "분석",
            "해석",
            "설명",
            "왜",
            "느려",
            "에러",
            "error",
            "explain",
            "optimize",
        ]
    )
    has_sql = _looks_like_sql(question) or any(
        _looks_like_sql(m.get("content", "")) for m in (history or []) if m.get("role") == "user"
    )
    if wants_analysis and has_sql:
        return "bigquery_sql_analysis"
    return "bigquery_sql_generation"


def _build_instruction_block(instruction_type: str) -> str:
    if instruction_type == "general_chat":
        return (
            """
[Directive]
너는 Slack에서 대화를 돕는 친절한 데이터 동료다.
일상 대화에는 자연스럽게 답하고, 데이터/쿼리 요청이 나오면 필요한 조건을 짧게 확인해라.
사실을 추정해 단정하지 말고, 모르면 모른다고 말하라.
데이터 요청이라면 "실행 가능한 쿼리 생성/검증/실행" 흐름으로 전환할 수 있게 필요한 조건을 물어라.

출력은 반드시 아래 JSON 스키마를 따를 것:
{
  "sql": "",
  "explanation": "<자연스러운 한국어 대화 응답>",
  "assumptions": [],
  "validation_steps": []
}
            """
        ).strip()
    if instruction_type == "bigquery_sql_generation":
        return (
            """
[Directive]
너는 BigQuery 표준 SQL을 작성하는 데이터 분석 도우미다. 목표는 "분석 의도를 검증 가능한 SQL로 구체화"하는 것이다.
            """
            + "\n\n"
            + _build_bigquery_sql_rules_block()
            + "\n\n"
            + _build_bigquery_sql_process_block()
        ).strip()
    if instruction_type == "bigquery_sql_analysis":
        return (
            """
[Directive]
당신은 실행 중심 데이터 분석 전문가다.
문제 진단뿐 아니라 실행 가능한 개선 SQL을 우선 제시한다.
결과는 반드시 아래 JSON 스키마를 따를 것.

동일한 출력 JSON 스키마:
{
  "sql": "<개선된 BigQuery SQL 또는 새 SQL 또는 빈 문자열>",
  "explanation": "<문제 진단, 개선 근거, 실행/비용 리스크>",
  "assumptions": "<가정 목록과 불확실성>",
  "validation_steps": [
    "스키마/키 유효성 점검",
    "작은 기간 샘플 실행",
    "엣지 케이스 검토"
  ]
}
성능/정확성/실행 가능성 지침은 생성 지시문과 동일하게 따른다.
            """
        ).strip()
    return (
        """
[Directive]
아래 JSON 스키마를 따르는 BigQuery 표준 SQL을 생성하라.
{
  "sql": "<BigQuery SQL>",
  "explanation": "<요청 해석 및 근거>",
  "assumptions": "<가정>",
  "validation_steps": ["체크리스트"]
}
        """
    ).strip()


def _clip_history(history: list[dict[str, Any]] | None, clip_limit: int) -> list[dict[str, Any]]:
    if not history:
        return []
    normalized: list[dict[str, Any]] = []
    for message in history:
        try:
            role = message.get("role")
            if role in ("user", "assistant"):
                normalized.append({"role": role, "content": message.get("content")})
        except Exception:
            continue
    if clip_limit > 0 and len(normalized) > clip_limit:
        return normalized[-clip_limit:]
    return normalized


def _build_laas_messages(
    *,
    question: str,
    table_info: str,
    glossary_info: str,
    history: list[dict[str, Any]] | None,
    images: list[dict[str, Any]] | None,
    instruction_type: str,
) -> list[dict[str, Any]]:
    context_block = (
        f"""
[Context]
DDL / Schema:
{table_info}

Glossary / Business Rules:
{glossary_info}
        """
    ).strip()
    instruction_block = _build_instruction_block(instruction_type)
    clip_limit = int(os.getenv("BIGQUERY_HISTORY_CLIP_LIMIT", "6"))
    normalized_history = _clip_history(history, clip_limit)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": context_block},
        {"role": "system", "content": instruction_block},
        *normalized_history,
        {"role": "user", "content": (question or "").strip()},
    ]
    if images:
        messages.append({"role": "user", "content": list(images)})
    return messages
