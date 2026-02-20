# Analyst Agent Service Refactor Plan (PydanticAI Best-Practice Aligned)

## 1) 목표
- `src/data_bolt/tasks/analyst_agent/service.py`의 과도한 책임(에이전트 생성, 출력 파싱, 메모리/컴팩션, 승인 흐름, 에러 복구)을 분리한다.
- 외부 계약(`run_analyst_turn`, `run_analyst_approval`, `has_thread_memory`)은 유지한다.
- 리팩토링 결과가 PydanticAI 공식 권장 패턴(의존성 주입, 도구 재사용, 복잡 워크플로 분리, 테스트 용이성)에 맞도록 구조를 재정의한다.

## 2) 공식 문서 기반 설계 원칙
1. 의존성 컨텍스트는 `deps` 객체로 주입하고, 비즈니스 로직에서 전역 상태 결합을 줄인다.
- 반영: `AnalystDeps` 중심으로 런타임 상태를 전달하고, helper 모듈이 `deps` 기반으로 동작.

2. 도구는 재사용 가능한 단위(toolset)로 묶어 조합 가능하게 관리한다.
- 반영: BigQuery/Schema 도구 등록 로직을 별도 모듈로 분리해 `service.py`에서 선언만 수행.

3. 복잡해지는 제어 흐름은 단일 에이전트 내부에 누적하지 않고, 함수/그래프 기반으로 외부 오케스트레이션한다.
- 반영: 승인 재개, 컴팩션, 복구 경로를 별도 실행 모듈로 분리해 `run_analyst_turn`은 orchestration facade로 축소.

4. 테스트는 모델/도구 실행을 오버라이드 가능한 구조로 설계한다.
- 반영: 출력 파서, 결과 매퍼, 복구 로직을 순수 함수 모듈로 분리하고 `service`는 wiring만 담당.

## 3) 타겟 구조
```text
src/data_bolt/tasks/analyst_agent/
  service.py                    # public facade only
  deps.py
  models.py
  tools.py
  runtime/
    __init__.py
    config.py                   # env/policy/deps builder
    output_parsing.py           # parsed output -> normalized fields
    result_contract.py          # base result/trace/output mapping
    thread_context.py           # history load/compaction/token ema
    recovery.py                 # retry/request_limit recovery
    approval_flow.py            # deferred approval resume path
```

## 4) 의존관계/위계 규칙 (인지 부하 최소화)
- `service.py -> runtime/*` 단방향 의존만 허용.
- `runtime/*`는 `service.py`를 import 금지.
- `runtime/*` 간에도 순환 의존 금지:
  - `config -> deps/models`
  - `output_parsing -> models(optional)`
  - `result_contract -> output_parsing, models`
  - `thread_context -> semantic_compaction/thread_memory_store`
  - `approval_flow -> result_contract, config, thread_context`
- `tools.py`는 계속 독립 유지(실제 tool 실행 책임).

## 5) 파일별 책임 분리 명세
1. `runtime/config.py`
- 시스템 프롬프트 상수
- env 기반 policy/deps 생성

2. `runtime/output_parsing.py`
- JSON/mapping/string output 정규화
- `AskUser/AnalystReply` 외 비정형 출력 안전 파싱

3. `runtime/result_contract.py`
- 결과 계약 생성 (`action`, `response_text`, `validation`, `execution`, `trace`)
- `DeferredToolRequests` -> approval response 블록 생성

4. `runtime/thread_context.py`
- thread key 계산
- memory load/save 보조
- compaction 후보 계산 및 token ema 업데이트

5. `runtime/recovery.py`
- tool retry 초과 복구
- request_limit 초과 복구

6. `runtime/approval_flow.py`
- approval context load/검증/deferred resume/delete
- approval 실행 후 memory append

## 6) 단계별 이행 순서
1. `runtime/output_parsing.py` 추출 및 테스트 고정
2. `runtime/result_contract.py` 추출
3. `runtime/thread_context.py` 추출
4. `runtime/recovery.py` 추출
5. `runtime/config.py` 추출
6. `runtime/approval_flow.py` 추출
7. `service.py` facade 정리(최종)

각 단계는 독립 커밋으로 진행하고, 단계 완료 전 다음 단계로 넘어가지 않는다.

## 7) 검증 계획
1. 계약 회귀
- `tests/tasks/test_analyst_agent_service.py`에서 주요 출력 필드 동일성 검증

2. CLI 회귀
- `tests/test_botctl_simulate.py`
- `tests/test_botctl_chat.py`
- trace가 런타임 trace만 노출되는지 확인

3. 정책/도구 회귀
- `tests/tasks/test_analyst_agent_tools.py`
- dry-run/approval/execute 정책 유지 확인

4. 품질 게이트
- `uv run ruff check .`
- `uv run pyright`
- `uv run pytest`

5. 수동 스모크
- `uv run python -m data_bolt.botctl.main simulate --text "안녕하세요" --trace`
- `uv run python -m data_bolt.botctl.main simulate --text "유저수 테이블이 뭐가 있는지 검색해줘" --trace`
- `uv run botctl chat --trace` (2턴 이상)

## 8) 비범위 (이번 리팩토링에서 하지 않음)
- 프롬프트 정책 자체 변경
- SQL 비용 정책 변경
- Slack UI/메시지 포맷 변경
- 새로운 툴 추가

## 9) 리스크와 대응
- 리스크: 모듈 분리 중 import cycle
  - 대응: `service.py`가 agent 인스턴스를 `approval_flow`에 주입하고, 역참조 금지
- 리스크: 비정형 output 파싱 동작 미세 변화
  - 대응: 기존 파싱 테스트 케이스 유지 + 회귀 fixture 보강
- 리스크: trace 누락
  - 대응: trace 노드 존재/순서 테스트를 필수 케이스로 유지

## 10) 참고한 공식 문서
- Agents: https://ai.pydantic.dev/agents/
- Toolsets: https://ai.pydantic.dev/toolsets/
- Multi-agent Applications: https://ai.pydantic.dev/multi-agent-applications/
- Graph Support: https://ai.pydantic.dev/graph/
- Testing: https://ai.pydantic.dev/testing/
