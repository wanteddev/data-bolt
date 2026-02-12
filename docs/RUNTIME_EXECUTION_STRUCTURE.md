# Runtime Execution Structure (Current)

이 문서는 **현재 코드 기준 실행 구조**를 정리한다.
정책 의도/설계 배경은 `docs/LANGGRAPH_BIGQUERY_BOT_SPEC.md`를 참고한다.

## 1) 엔트리포인트

### A. Slack 실서비스
1. Slack -> `POST /slack/events` (`/Users/woojing/code/wanted/data-bolt/src/data_bolt/app.py`)
2. Bolt handler ack 후 background invoke (`/Users/woojing/code/wanted/data-bolt/src/data_bolt/slack/handlers.py`)
3. Background Lambda -> `POST /slack/background`
4. `handle_bigquery_sql_bg()` -> `run_bigquery_agent()`
5. LangGraph 실행 후 Slack thread 응답 전송

### B. botctl simulate/chat
1. `simulate`/`chat` 명령이 `run_bigquery_agent()`를 직접 호출
2. `--via-background`는 `/slack/background` 경로를 로컬 재현

## 2) BigQuery Agent 그래프 (thread 단위)

그래프 정의: `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery_agent/graph.py`

노드/분기:
1. `reset_turn_state`
2. `classify_relevance`
3. relevance=false -> `clear_response` -> `END`
4. relevance=true -> `ingest` -> `plan_turn_action`
5. `plan_turn_action` 결과(`action`)에 따라 분기
- `chat_reply` -> `chat_reply` -> `compose_response`
- `schema_lookup` -> `schema_lookup` -> `compose_response`
- `sql_validate_explain` -> `sql_validate_explain` -> `compose_response`
- `sql_generate` -> `validate_candidate_sql` -> `policy_gate`
- `sql_execute` -> `validate_candidate_sql` -> `policy_gate`
- `execution_approve`/`execution_cancel` -> `policy_gate`
6. `policy_gate`
- `can_execute=true` -> `execute_sql` -> `compose_response`
- 아니면 `compose_response`
7. `END`

핵심 포인트:
- 의도(intent) 대신 **단일 액션(action)** 라우팅을 사용한다.
- `schema_lookup`, `sql_validate_explain`는 실행 경로로 가지 않는다.
- 실제 실행은 `policy_gate` 통과 시점에만 가능하다.

## 3) 액션 라우터 계약

라우터 함수: `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery/service.py::plan_turn_action`

출력 스키마:
- `action`: `chat_reply | schema_lookup | sql_validate_explain | sql_generate | sql_execute | execution_approve | execution_cancel`
- `confidence`: float
- `reason`: string

입력 컨텍스트:
- 최신 사용자 메시지
- 최근 대화 히스토리
- pending execution SQL 존재 여부
- 이전 SQL/dry-run 상태
- 사용자 SQL 블록 포함 여부

실패/예외 시 정책:
- 규칙 기반 의도 추론으로 되돌아가지 않음
- 안전 채팅 폴백(`chat_reply`) 사용

## 4) 액션별 책임

- `chat_reply`: 자유 대화 응답 생성(도구 실행 없음)
- `schema_lookup`: 스키마/RAG 설명 + 참고 SQL 제시 가능(실행 금지)
- `sql_validate_explain`: 사용자 SQL/직전 SQL dry-run 검증 + 설명(실행 금지)
- `sql_generate`: SQL 생성(+검증) 후 실행 정책 판단
- `sql_execute`: 실행 대상 SQL 선택/검증 후 실행 정책 판단
- `execution_approve`/`execution_cancel`: 승인 상태만 처리, 최종 강제는 `policy_gate`

## 5) 실행 정책(`policy_gate`)

구현: `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery_agent/nodes.py::_node_policy_gate`

검사 항목:
1. read-only 단일 statement 여부
2. dry-run 존재/성공 여부
3. `BIGQUERY_MAX_BYTES_BILLED` 제한
4. 비용 임계값(`BIGQUERY_AUTO_EXECUTE_MAX_COST_USD`) 비교
5. 승인 대기/승인/취소 상태

결과:
- `auto_execute`
- `approval_required`
- `blocked`

## 6) SQL 생성 내부 워크플로

`build_bigquery_sql()` 내부는 선택적으로 내부 workflow graph를 사용한다.

- 구현: `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery/workflow_graph.py`
- 플래그: `BIGQUERY_SQL_WORKFLOW_GRAPH_ENABLED`
- 역할: SQL 생성 -> dry-run -> refine 루프

## 7) 체크포인터/캐시

구현: `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery_agent/checkpoint.py`

- `memory`: in-memory saver + compiled graph 재사용
- `postgres`: 연결별 compiled graph/context 캐시
- `dynamodb`: table/region/endpoint 키별 compiled graph 캐시

정책:
- `dynamodb`는 fail-fast
- `postgres`는 설정/연결 계열 오류에서만 optional memory fallback 허용

## 8) 변경 시 동기화 파일

- 그래프/노드 계약 변경:
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery_agent/graph.py`
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery_agent/nodes.py`
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery_agent/types.py`
- 서비스 계약 변경:
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery/service.py`
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery_agent/service.py`
- 문서 동기화:
- `/Users/woojing/code/wanted/data-bolt/docs/RUNTIME_EXECUTION_STRUCTURE.md`
- `/Users/woojing/code/wanted/data-bolt/docs/LANGGRAPH_BIGQUERY_BOT_SPEC.md`

---
Last updated: 2026-02-11
