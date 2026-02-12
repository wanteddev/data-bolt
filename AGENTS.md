# AGENTS.md

이 저장소는 AWS Lambda에서 동작하는 Slack 봇 애플리케이션입니다. Litestar + Slack Bolt 기반이며 Lambda Web Adapter를 사용해 단일 컨테이너 이미지로 여러 함수를 운영합니다.

## 아키텍처 요약
- **Web Function**: Function URL로 HTTP 요청 처리
- **Slack Background Function**: 장시간 Slack 작업 비동기 처리
- 모든 함수가 동일한 이미지(단일 Docker 이미지)를 공유하고 `AWS_LWA_PASS_THROUGH_PATH`로 라우팅됩니다.

## 핵심 경로
- `src/data_bolt/app.py`: Litestar 앱 엔트리
- `src/data_bolt/slack/`: Slack Bolt 앱, 핸들러, 백그라운드 처리
- `src/data_bolt/tasks/bigquery/`: BigQuery SQL 생성/검증/LLM/RAG 모듈
- `src/data_bolt/tasks/bigquery_agent/`: LangGraph 에이전트 노드/그래프/체크포인트/서비스
- `template.yaml`: CloudFormation 스택 정의
- `scripts/`: 배포/운영 스크립트 (`build_image.sh`, `deploy_stack.sh`, `sync_env.py`, `sync_secrets_to_ssm.sh` 등)
- `justfile`: 개발/배포 커맨드 모음

## 코딩 가이드 (Litestar)
- 라우트 핸들러는 기본적으로 `async`로 작성합니다.
- 동기 핸들러가 필요하면 `sync_to_thread`를 명시해 thread pool 실행 여부를 선언합니다.
- `boto3`, `slack_sdk`처럼 동기 I/O 라이브러리는 `anyio.to_thread.run_sync`로 오프로딩합니다.

## Tasks 모듈 원칙
- `src/data_bolt/tasks/`는 **순수 비즈니스 로직의 동기 함수**만 둡니다.
- tasks 내부에서는 **Litestar/Slack 같은 런타임 의존성**을 가져오거나 **async 함수**를 정의하지 않습니다.
- tasks는 **호출 당하는 존재**로 유지합니다. 런타임 어댑터(handlers/background)에서 필요 시 `anyio.to_thread.run_sync`로 호출합니다.
- 간단한 Slack DM/슬래시 커맨드 처리는 handlers에 직접 구현하고, **3초 이상 소요될 가능성이 있을 때만** SlackBgFunction으로 분리합니다.

## LangGraph 구현 가이드
- 상태는 `thread` 단위 지속 상태와 `turn` 단위 임시 상태를 분리합니다. `candidate_sql`, `dry_run`, `execution`, `response_text` 같은 임시 상태는 그래프 시작 노드에서 매 턴 명시적으로 초기화합니다.
- relevance 판단 전에는 대화 히스토리에 신규 user 메시지를 append하지 않습니다. `should_respond=False`인 메시지는 모델 컨텍스트를 오염시키지 않아야 합니다.
- `should_respond=False` 분기는 `END`로 바로 가지 말고, `response_text=""`를 보장하는 정리 노드를 거쳐 종료합니다.
- 체크포인터 정책: `dynamodb`는 fail-fast(오류 시 fallback 없음), `postgres`는 연결/설정 오류에서만 `memory` fallback을 허용합니다.
- 체크포인터(saver)와 compiled graph는 warm runtime에서 재사용합니다. 요청마다 `compile()`/연결 재생성을 하지 않습니다.
- LangGraph 변경 시 회귀 테스트에 최소 2턴 시나리오를 포함합니다. (예: 이전 턴 SQL/응답이 다음 턴에 누수되지 않는지, ignore 턴이 conversation 길이를 늘리지 않는지)

## LLM 구현 원칙
- 하드코딩된 규칙/패턴 매칭으로 LLM 출력을 강제 보정하는 방식은 지양합니다.
- SQL 품질/호환성/안전성은 우선적으로 프롬프트 지시문과 컨텍스트(DDL/용어집/에러 피드백) 설계를 통해 해결합니다.
- 코드 레벨 제약은 모델 실패 시 시스템 안전을 위한 최소 범위(예: 실행 가드, 재시도 상한, fail-fast)에만 사용합니다.
- 특정 문법 금지/권장 사항이 필요하면 파서/정규식 하드코딩보다 프롬프트 규칙으로 먼저 반영하고, 회귀 테스트로 검증합니다.

## Slack 롱러닝 처리 (Lambda + LWA)
- Slack 요청은 3초 내 2xx/ack 응답이 원칙입니다. 이벤트는 빠르게 200 응답 후 실제 처리를 분리합니다.
- 이 템플릿은 WebFunction이 수신 후 `SlackBgFunction`으로 작업을 위임하는 흐름을 기본으로 합니다.
- 후속 응답은 `response_url` 또는 Slack Web API(`chat.postMessage`)로 전송합니다.
- FaaS 환경에서 `process_before_response=True`를 사용하는 경우, 리스너는 3초 내 종료되어야 하므로 장시간 작업은 반드시 분리합니다.

## Project Structure & Module Organization
- `src/data_bolt/`: Litestar app entry (`app.py`) and Slack Bolt integration in `slack/` (`app.py`, `handlers.py`, `background.py`).
- `src/data_bolt/tasks/bigquery/`: BigQuery SQL 도메인 모듈 (`service.py`, `execution.py`, `llm_client.py` 등).
- `src/data_bolt/tasks/bigquery_agent/`: LangGraph 오케스트레이션 모듈 (`nodes.py`, `graph.py`, `checkpoint.py`, `service.py`).
- `tests/`: pytest suite (`test_*.py`, `conftest.py`).
- `scripts/`: deployment and automation helpers (build, deploy, env sync, secrets sync).
- `docs/`: operational notes (e.g., Slack guide).
- Infra/config: `template.yaml` (CloudFormation), `Dockerfile`, `justfile`, `dot_env.example`.

## Build, Test, and Development Commands
- `uv sync --frozen`: install dependencies from `uv.lock`.
- `just serve`: run the Litestar dev server on `http://localhost:8080`.
- `just build [tag]`: build the container image (tag defaults to timestamp).
- `just deploy [tag]`: push to ECR and deploy CloudFormation.
- `AWS_PROFILE=sandbox STACK_NAME=data-bolt just url`: fetch the deployed Function URL.
- `just sync-env`: sync `LAMBDA_*` vars from `.env` into `template.yaml` and deploy scripts.
- `just sync-secrets`: sync Slack secrets to SSM SecureString parameters.
- `uv run pytest`: run tests with coverage; HTML report goes to `htmlcov/`.

## 배포/시크릿 정책
- 비밀값(`SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`)은 SSM SecureString 경로(`/data-bolt/slack/bot-token`, `/data-bolt/slack/signing-secret`)에 저장합니다.
- `just deploy`는 위 SSM 경로에서 비밀값을 읽어 CloudFormation `NoEcho` 파라미터로 주입합니다.
- 비밀값 변경 시 `AWS_PROFILE=sandbox AWS_REGION=ap-northeast-2 just sync-secrets` 후 배포합니다.
- 비밀이 아닌 운영 설정값(예: `LANGGRAPH_CHECKPOINT_BACKEND`)은 환경변수/CloudFormation 파라미터로 관리합니다.
- `LANGGRAPH_CHECKPOINT_BACKEND`를 바꿀 때는 배포 시 `LANGGRAPH_CHECKPOINT_BACKEND=<value> just deploy`를 사용합니다.

## 이미지 빌드 정책
- Lambda 호환성을 위해 `scripts/build_image.sh`는 `docker buildx build --provenance=false --sbom=false --load --platform linux/arm64`를 사용합니다.
- `scripts/deploy_stack.sh`는 ECR `imageManifestMediaType`을 검사하고, Lambda 비호환 포맷(OCI index/manifest list)이면 배포를 중단합니다.

## 환경 변수 / Security & Configuration Tips
- `dot_env.example`을 복사해 `.env`를 만들고 값을 채웁니다.
- 새 Lambda 환경변수는 `.env`에 `LAMBDA_` prefix로 추가한 뒤 `just sync-env`로 동기화합니다.
- 실행 자동화 비용 임계값은 `LAMBDA_BIGQUERY_AUTO_EXECUTE_MAX_COST_USD`로 제어합니다. (기본 `1.0`, 미산출 비용은 승인 필요)
- 별도 AWS 프로파일이 필요하면 `AWS_PROFILE=프로파일명 <command>` 형태로 실행합니다.
- 기본 배포 타깃: `AWS_PROFILE=sandbox`, `STACK_NAME=data-bolt`, `AWS_REGION=ap-northeast-2`.
- 로컬에서만 비밀값을 관리하고, 레포에 커밋하지 않습니다.

## Coding Style & Naming Conventions
- Python 3.13, source under `src/` (module name `data_bolt`).
- Formatting via Ruff (`line-length = 100`, double quotes). Run `uv run ruff format`.
- Linting via Ruff rules `B`, `I`, `RUF`, `UP` and `uv run ruff check --fix`.
- Type checking via strict pyright: `uv run pyright`.
- Use `snake_case` for functions/variables; tests named `test_*.py`.

## 테스트/품질
```bash
uv run ruff check .
uv run ruff format .
uv run pyright
uv run pytest
```

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio`, `pytest-cov`.
- Markers available: `unit`, `integration`, `slow` (e.g., `uv run pytest -m unit`).
- Prefer small, focused tests in `tests/` that mirror module names.

## botctl simulate 테스트 가이드
- `simulate`는 Slack/AWS 배포 없이 로컬에서 에이전트 라우팅/생성/검증 흐름을 재현합니다.
- 기본 `.env`를 로드하며, 다른 파일을 쓰려면 `--env-file`을 사용합니다.
- JSON 결과 확인은 `--json --no-trace` 조합을 권장합니다.

```bash
# 1) 단일 입력(기본)
uv run python -m data_bolt.botctl.main simulate --text "안녕하세요"

# 2) JSON 출력(자동화/디버깅 용도)
uv run python -m data_bolt.botctl.main simulate --text "어제 가입자 수를 일자별로 알려줘" --json --no-trace

# 3) 기본 케이스 실행
uv run python -m data_bolt.botctl.main simulate --case sql_gen --json --no-trace

# 4) 케이스 파일 실행
uv run python -m data_bolt.botctl.main simulate --file tests/fixtures/botctl_cases.json --json --no-trace

# 5) 백그라운드 경로(/slack/background) 재현
uv run python -m data_bolt.botctl.main simulate --text "지난주 가입자 쿼리 만들어줘" --via-background --json --no-trace
```

- 멀티턴 재현:
  - `--thread-ts`를 같은 값으로 반복 실행하면 thread 단위 문맥을 재현할 수 있습니다.
  - `memory` backend에서는 `--thread-ts`가 지원되지 않습니다. (`postgres`/`dynamodb` backend 필요)

```bash
# 예시: postgres backend에서 멀티턴
LANGGRAPH_CHECKPOINT_BACKEND=postgres \
uv run python -m data_bolt.botctl.main simulate --text "지난주 가입자 수 쿼리 만들어줘" --thread-ts demo-1 --json --no-trace

LANGGRAPH_CHECKPOINT_BACKEND=postgres \
uv run python -m data_bolt.botctl.main simulate --text "방금 쿼리 실행해줘" --thread-ts demo-1 --json --no-trace
```

- LLM provider/모델 교차 검증:
  - `.env`를 바꾸지 않고 실행 시점 override로 빠르게 비교할 수 있습니다.

```bash
# 예시: OpenAI-compatible + glm-4.5-air로 단발 테스트
LLM_PROVIDER=openai_compatible \
LLM_OPENAI_BASE_URL=https://api.z.ai/api/coding/paas/v4 \
LLM_OPENAI_MODEL=glm-4.5-air \
LLM_TIMEOUT_SECONDS=120 \
LLM_TIMEOUT_GENERATION_SECONDS=120 \
uv run python -m data_bolt.botctl.main simulate --text "어제 가입자 수를 일자별로 알려줘" --json --no-trace
```

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and can be English or Korean (no enforced prefix).
- PRs should include: summary, test results/commands, and any config or deployment impact.
- If Slack message formats change, include before/after examples in the PR.

## Agent-Specific Instructions
- When using a specific AWS CLI profile, always prefix commands with `AWS_PROFILE=<profile>`.
- Default deployment targets for this repo: `AWS_PROFILE=sandbox`, `STACK_NAME=data-bolt`, `AWS_REGION=ap-northeast-2`.

## 문서
- `docs/SLACK_GUIDE.md`: 메시지 포맷, 실패 대응, 트러블슈팅
- `docs/RUNTIME_EXECUTION_STRUCTURE.md`: 현재 코드 기준 실행 경로/그래프/분기 구조(구조 분석 시 우선 참조)

## 실행 구조 문서화 규칙
- 실행 구조를 분석하거나 설명할 때는 먼저 `docs/RUNTIME_EXECUTION_STRUCTURE.md`를 기준으로 확인합니다.
- 실행 경로(엔트리포인트, 라우팅, 그래프 노드/엣지, checkpoint fallback, simulate 경로)가 변경되면 코드 변경과 함께 `docs/RUNTIME_EXECUTION_STRUCTURE.md`를 같은 PR/커밋에서 업데이트합니다.
- 문서와 코드가 불일치하면 코드를 기준으로 즉시 문서를 수정하고, 리뷰/공유 시 수정된 문서 기준으로 설명합니다.
