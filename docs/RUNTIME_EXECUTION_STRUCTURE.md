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

## 2) BigQuery Agent 실행 구조 (thread 단위)

구현:
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery_agent/loop_runtime.py`

토폴로지:
1. `START`
2. `prepare_node`
3. `agent_node`
4. `tools_node`
5. `finalize_node`
6. `END`

핵심 흐름:
1. `prepare_node`: turn 초기화 + 통합 결정기(`decide_turn`)의 relevance-only 판정 + ingest
2. `agent_node`: 통합 결정기(`decide_turn`)가 `action/intent_mode/execution_intent/turn_mode/planned_tool`을 결정하고, 필요 시 SQL 생성(`sql_generate`)
3. `agent_node` 조건부 분기
- tool_calls 있음 -> `tools_node`
- tool_calls 없음 -> `finalize_node`
4. `tools_node`: LLM 없이 tool registry를 통해 기계적으로 실행 후 `agent_node`로 복귀
5. `tools_node`는 실행 가드/HITL를 집행하며 승인 대기 시 `execution.status=pending_approval` 및 `execution.request`를 기록
6. `agent_node <-> tools_node` 루프는 `turn.max_steps`까지 반복 가능
7. `finalize_node`: `compose_response` 수행 후 종료

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

추가 메타:
- `turn_mode`: `chat | analyze | execute`
- `planned_tool`: action 기준으로 선택된 tool 이름(없으면 빈 문자열)
- `intent_mode`: `analysis | retrieval | execution | chat`
- `execution_intent`: `none | suggested | explicit`
- `needs_clarification`: 모호성 질문 필요 여부

## 4) 액션별 책임

- `chat_reply`: agent가 최종 자연어 응답 생성(도구 실행 없음)
- `schema_lookup`: agent가 `schema_lookup(...)` tool_call 생성, tools가 실행
- `sql_validate_explain`: agent가 `sql_validate_explain(sql=...)` tool_call 생성, tools가 dry-run 검증 수행
- `sql_generate`: agent가 SQL 생성(LLM)과 검증 정보를 제공한다. 실행은 `execution_intent=explicit`일 때만 후속 `sql_execute`로 진입한다.
- `sql_execute`: agent가 대상 SQL args를 구성, tools가 guard + 실행/보류 처리
- `execution_approve`/`execution_cancel`: agent가 승인/취소 tool_call 생성, tools가 pending request에 대해 집행

## 5) 실행 정책

구현:
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery/tools.py::GuardedExecuteTool.run`
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery_agent/loop_runtime.py::tools_node`

정책 집행:
1. read-only 단일 statement 여부
2. dry-run 존재/성공 여부
3. `BIGQUERY_MAX_BYTES_BILLED` 제한
4. 비용 임계값(`BIGQUERY_AUTO_EXECUTE_MAX_COST_USD`) 비교
5. 승인/취소 상태

핵심:
- 실제 BigQuery execute 호출은 `GuardedExecuteTool` 내부 단일 경로에서만 허용
- DML/DDL은 승인 여부와 무관하게 무조건 차단
- 승인 필요 시 tools 노드는 실제 실행 대신 `PENDING_APPROVAL` 상태를 기록하고 종료한다.

## 6) SQL 생성 내부 워크플로

`build_bigquery_sql()` 내부는 선택적으로 내부 workflow graph를 사용한다.

- 구현: `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery/workflow_graph.py`
- 플래그: `BIGQUERY_SQL_WORKFLOW_GRAPH_ENABLED`
- 역할: SQL 생성 -> dry-run -> refine 루프

## 7) 체크포인터/캐시

구현: `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery_agent/loop_runtime.py`

- 백엔드:
  - `memory`: in-memory saver + compiled graph 재사용
  - `postgres`: 연결별 compiled graph/context 캐시
  - `dynamodb`: table/region/endpoint 키별 compiled graph 캐시

정책:
- `dynamodb`는 fail-fast
- `postgres`는 설정/연결 계열 오류에서만 optional memory fallback 허용

## 8) 변경 시 동기화 파일

- 그래프/노드 계약 변경:
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery_agent/loop_runtime.py`
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery_agent/nodes.py`
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery_agent/types.py`
- 서비스 계약 변경:
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery/service.py`
- `/Users/woojing/code/wanted/data-bolt/src/data_bolt/tasks/bigquery_agent/service.py`
- 문서 동기화:
- `/Users/woojing/code/wanted/data-bolt/docs/RUNTIME_EXECUTION_STRUCTURE.md`
- `/Users/woojing/code/wanted/data-bolt/docs/LANGGRAPH_BIGQUERY_BOT_SPEC.md`

---
Last updated: 2026-02-12
