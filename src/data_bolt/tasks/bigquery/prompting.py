"""Prompt and message construction helpers."""

from __future__ import annotations

import os
from typing import Any

from .parser import _looks_like_sql


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
너는 BigQuery 표준 SQL을 작성하는 데이터 엔지니어 도우미다.
반드시 아래 규칙과 절차를 따른다.

[응답 규칙]
1) BigQuery 표준 SQL만 사용 (legacy 금지), SELECT * 금지, 스키마에 없는 컬럼 추측 금지.
2) 한국 시간(Asia/Seoul) 고려 시 TIMESTAMP/DATETIME 변환을 명시.
3) 조인 시 키와 null 처리 근거를 설명에 적시.
4) 결과는 단일 쿼리 또는 CTE로 제공. 임시 테이블 금지.
5) 출력 JSON 스키마:
{
  "sql": "<BigQuery SQL>",
  "explanation": "<요청 해석, 조인/필터 근거, 시간대 처리 근거>",
  "assumptions": "<제공되지 않은 가정 목록 또는 빈 배열/문자열>",
  "validation_steps": [
    "스키마 존재 확인 방법",
    "작은 기간으로 샘플 실행해 행수/NULL 비율 검증",
    "엣지 케이스 점검 아이디어"
  ]
}
6) COUNT(_) 금지, COUNT(*) 사용. 7) 필요한 컬럼만 SELECT.

[절차]
1) 요청 재진술 → 2) 컨텍스트 선택 → 3) 설계 설명 → 4) SQL 생성 → 5) 자체 검증.
            """
        ).strip()
    if instruction_type == "bigquery_sql_analysis":
        return (
            """
[Directive]
당신은 데이터 분석 전문가로써 요청에 맞게 응답을 제공한다.
답변은 언제나 읽기 쉽고 이해하기 쉬워야 하며 질문자의 의도에 부합해야 한다.
최대한 질문자를 도와주기 위해서 최선을 다해야 합니다.
당신은 차가운 기계가 아닌 질문자의 최선을 다해서 도우려는 따뜻한 동료입니다.
결과는 반드시 아래 JSON 스키마를 따를 것.

동일한 출력 JSON 스키마:
{
  "sql": "<개선된 BigQuery SQL 또는 새 SQL 또는 빈 문자열>",
  "explanation": "<문제 진단 및 개선 근거>",
  "assumptions": "<가정 목록>",
  "validation_steps": [
    "스키마/키 유효성 점검",
    "작은 기간 샘플 실행",
    "엣지 케이스 검토"
  ]
}
성능/정확성 지침은 생성 지시문과 동일하게 따른다.
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
