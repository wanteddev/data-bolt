# LangGraph BigQuery Bot Spec (Current)

## 1) 목적

이 문서는 현재 구현 기준으로 BigQuery 봇의 라우팅/실행 정책을 정의한다.
핵심 방향은 다음 3가지다.

1. 하드코딩 키워드 의도판단 제거
2. 단일 액션 라우터 기반 노드 선택
3. 실행 경로와 비실행 경로(`schema_lookup`, `sql_validate_explain`)의 명확한 분리
4. 분석 의도 구체화(clarification) 우선, 실행은 명시 의도 기반으로 분리

## 2) 결정사항

1. 라우팅 모델: **Single-Action Router**
2. 런타임 모델: **Loop 단일 런타임**
3. `schema_lookup`: RAG 설명 + 참고 SQL 허용, 실행 금지
4. 라우터 실패 시: 규칙 기반 추론으로 회귀하지 않고 안전 채팅 폴백

## 3) 액션 계약

`TurnAction`:
- `ignore`
- `chat_reply`
- `schema_lookup`
- `sql_validate_explain`
- `sql_generate`
- `sql_execute`
- `execution_approve`
- `execution_cancel`

`AgentResult` 계약:
- `intent` 제거, `action` 사용
- `routing` 메타는 `action`, `confidence`, `reason`, `route`, `fallback_used` 중심

## 4) 런타임 구조

구현 파일:
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery_agent/loop_runtime.py`
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery_agent/nodes.py`

- `prepare -> agent <-> tools -> finalize` 토폴로지를 사용한다.
- SQL 실행 허용 판정과 실제 실행은 `guarded_execute_tool`에서 단일 관문으로 집행한다.

## 5) 라우터 프롬프트 계약

구현 함수:
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery/service.py::plan_turn_action`

입력 컨텍스트:
- 최신 사용자 메시지
- 최근 대화 히스토리
- pending execution 여부
- 최근 SQL/dry-run 보유 여부
- 사용자 SQL 블록 포함 여부

출력(JSON 강제):
- `action`
- `intent_mode`
- `needs_clarification`
- `clarifying_question`
- `execution_intent`
- `confidence`
- `reason`

규칙:
- 스키마/가능 데이터 문의 -> `schema_lookup`
- SQL 검증/설명 요청 -> `sql_validate_explain`
- 데이터 조회/집계 요청 -> `sql_generate` (단, 모호하면 clarification 우선)
- 명시 실행/승인/취소 -> `sql_execute` / `execution_approve` / `execution_cancel`
- 불확실/실패 시 -> `chat_reply` 폴백

## 6) 실행 정책

구현:
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery/tools.py::GuardedExecuteTool.run`

적용 범위:
- `sql_execute`, `execution_approve`, `execution_cancel`

검사 항목:
1. read-only SQL 여부
2. dry-run 성공 여부
3. 바이트 제한(`BIGQUERY_MAX_BYTES_BILLED`)
4. 비용 임계값(`BIGQUERY_AUTO_EXECUTE_MAX_COST_USD`)
5. 승인/취소 상태

공통 규칙:
- `schema_lookup`, `sql_validate_explain`에서는 execute를 절대 호출하지 않는다.
- `sql_generate`는 기본적으로 실행을 트리거하지 않는다. `execution_intent=explicit`일 때만 실행 단계로 진입한다.
- DML/DDL은 승인 여부와 무관하게 무조건 차단한다.
- 비용 임계값 초과 또는 비용 미산출 시 HITL 승인(`실행 승인`/`실행 취소`)을 요구한다.

## 7) 자유대화와 도구 사용

- 자유대화 자체는 `chat_reply`에서 처리한다.
- 전체 턴 라우팅은 single-action으로 동작하며, 필요 시
  `schema_lookup` 또는 `sql_validate_explain`을 직접 선택할 수 있다.

## 8) 서비스 API

- `classify_intent` -> `plan_turn_action`로 대체
- `plan_free_chat`는 대화 응답 생성 전용
- `explain_schema_lookup`, `explain_sql_validation` 제공

관련 파일:
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery/service.py`
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery/__init__.py`
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery_agent/service.py`

## 9) 테스트 기준

필수 검증:
1. `schema_lookup`은 실행하지 않는다.
2. `sql_validate_explain`은 dry-run 설명만 수행하고 실행하지 않는다.
3. `sql_generate`는 생성/검증/정책 게이트를 유지한다.
4. `execution_approve/cancel`이 정상 처리된다.
5. 라우터 실패 시 `chat_reply` 안전 폴백이 동작한다.
6. ignore 턴은 conversation을 오염시키지 않는다.
7. 실행/비실행 턴 모두 다음 분석 제안(2~3개)을 제공하고 thread 단위 analysis brief를 갱신한다.

## 10) 문서 동기화 규칙

실행 경로/노드/상태 계약이 변경되면 다음 문서를 같은 변경셋에서 함께 수정한다.

- `/Users/woojing/code/wanted/data-bolt/docs/RUNTIME_EXECUTION_STRUCTURE.md`
- `/Users/woojing/code/wanted/data-bolt/docs/LANGGRAPH_BIGQUERY_BOT_SPEC.md`

---
Last updated: 2026-02-12
