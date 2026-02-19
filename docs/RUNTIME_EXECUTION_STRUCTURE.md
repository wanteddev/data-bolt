# Runtime Execution Structure (Current)

이 문서는 **현재 코드 기준** Data Bolt의 실행 경로를 정리합니다.

## 1) 엔트리포인트

### A. Slack 실서비스
1. Slack -> `POST /slack/events` (`src/data_bolt/app.py`)
2. Bolt handler가 3초 내 `ack()` 수행 후 background invoke (`src/data_bolt/slack/handlers.py`)
3. Background Lambda -> `POST /slack/background`
4. `task_type=bigquery_sql|build_bigquery`는 `handle_bigquery_sql_bg()`로 라우팅
5. `task_type=bigquery_approval`는 `handle_bigquery_approval_bg()`로 라우팅
6. 결과를 Slack thread에 메시지/버튼으로 전송

### B. botctl simulate/chat
- `botctl simulate --direct`는 `run_analyst_turn()`을 직접 호출
- `botctl simulate --via-background`는 `/slack/background` 경로를 로컬 재현
- `botctl chat`는 동일한 direct 경로를 반복 호출해 다턴 테스트

## 2) Analyst Agent 런타임

구현:
- `src/data_bolt/tasks/analyst_agent/service.py`
- `src/data_bolt/tasks/analyst_agent/tools.py`

핵심 구성:
- `run_analyst_turn(payload)`
- `run_analyst_approval(payload)`
- PydanticAI `Agent` + tool 3종
  - `get_schema_context`
  - `bigquery_dry_run`
  - `bigquery_execute`

정책:
- `tasks` 내부는 동기 함수만 사용
- `Slack/background`에서 `anyio.to_thread.run_sync`로 오프로딩
- UsageLimits로 runaway 호출 제한

## 3) 툴 체인 규칙

실행 순서:
1. `get_schema_context` (필요 시)
2. `bigquery_dry_run`
3. `bigquery_execute`

강제 정책:
- 실행 전 dry-run 필수
- DML/DDL/non-SELECT 또는 고비용 쿼리는 `ApprovalRequired`
- 하드 리밋 초과 시 `ModelRetry`로 쿼리 수정 유도

## 4) 승인(Deferred Tool) 흐름

구현:
- `src/data_bolt/tasks/analyst_agent/approval_store.py`
- `src/data_bolt/slack/handlers.py` action handlers

흐름:
1. `bigquery_execute`가 `ApprovalRequired`를 던지면 `DeferredToolRequests` 생성
2. `run_analyst_turn()`이 승인 컨텍스트를 DynamoDB TTL에 저장
3. Slack thread에 `실행 승인` / `실행 취소` 버튼 노출
4. 버튼 클릭(`bq_approve_execute` / `bq_deny_execute`) 시 `task_type=bigquery_approval` invoke
5. `run_analyst_approval()`이 `DeferredToolResults`로 재실행

저장 항목:
- `approval_request_id` (PK)
- `tool_call_ids`, `deferred_metadata`
- `session_messages_json`
- `requester/channel/thread/team`
- `expires_at` (TTL)

## 5) 히스토리 정책

- 일반 대화 히스토리: Slack thread 메시지를 매 턴 fetch 후 `message_history`로 주입
- 승인 컨텍스트: DynamoDB에 최소 저장하여 버튼 승인 시 동일 컨텍스트 재개

## 6) LLM Provider

구현:
- `src/data_bolt/tasks/analyst_agent/model_factory.py`

지원:
- `LLM_PROVIDER=openai_compatible` -> OpenAI-compatible provider
- `LLM_PROVIDER=laas` -> LAAS `/api/preset/v2/chat/completions` FunctionModel adapter

## 7) 변경 시 동기화 대상

아래 파일 변경 시 본 문서를 함께 업데이트합니다.
- `src/data_bolt/tasks/analyst_agent/service.py`
- `src/data_bolt/tasks/analyst_agent/tools.py`
- `src/data_bolt/slack/background.py`
- `src/data_bolt/slack/handlers.py`

---
Last updated: 2026-02-19
