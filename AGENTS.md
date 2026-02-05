# AGENTS.md

이 저장소는 AWS Lambda에서 동작하는 Slack 봇 애플리케이션입니다. Litestar + Slack Bolt 기반이며 Lambda Web Adapter를 사용해 단일 컨테이너 이미지로 여러 함수를 운영합니다.

## 아키텍처 요약
- **Web Function**: Function URL로 HTTP 요청 처리
- **Slack Background Function**: 장시간 Slack 작업 비동기 처리
- 모든 함수가 동일한 이미지(단일 Docker 이미지)를 공유하고 `AWS_LWA_PASS_THROUGH_PATH`로 라우팅됩니다.

## 핵심 경로
- `src/data_bolt/app.py`: Litestar 앱 엔트리
- `src/data_bolt/slack/`: Slack Bolt 앱, 핸들러, 백그라운드 처리
- `template.yaml`: CloudFormation 스택 정의
- `scripts/`: 배포/운영 스크립트 (`create_roles.sh`, `deploy_stack.sh`, `sync_env.py` 등)
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

## Slack 롱러닝 처리 (Lambda + LWA)
- Slack 요청은 3초 내 2xx/ack 응답이 원칙입니다. 이벤트는 빠르게 200 응답 후 실제 처리를 분리합니다.
- 이 템플릿은 WebFunction이 수신 후 `SlackBgFunction`으로 작업을 위임하는 흐름을 기본으로 합니다.
- 후속 응답은 `response_url` 또는 Slack Web API(`chat.postMessage`)로 전송합니다.
- FaaS 환경에서 `process_before_response=True`를 사용하는 경우, 리스너는 3초 내 종료되어야 하므로 장시간 작업은 반드시 분리합니다.

## Project Structure & Module Organization
- `src/data_bolt/`: Litestar app entry (`app.py`) and Slack Bolt integration in `slack/` (`app.py`, `handlers.py`, `background.py`).
- `tests/`: pytest suite (`test_*.py`, `conftest.py`).
- `scripts/`: deployment and automation helpers (build, deploy, env sync, IAM roles).
- `docs/`: operational notes (e.g., Slack guide).
- Infra/config: `template.yaml` (CloudFormation), `Dockerfile`, `justfile`, `dot_env.example`.

## Build, Test, and Development Commands
- `uv sync --frozen`: install dependencies from `uv.lock`.
- `just serve`: run the Litestar dev server on `http://localhost:8080`.
- `just build [tag]`: build the container image (tag defaults to timestamp).
- `just deploy [tag]`: push to ECR and deploy CloudFormation.
- `AWS_PROFILE=sandbox STACK_NAME=data-bolt just url`: fetch the deployed Function URL.
- `just create-roles`: create IAM roles for the stack.
- `just sync-env`: sync `LAMBDA_*` vars from `.env` into `template.yaml` and deploy scripts.
- `uv run pytest`: run tests with coverage; HTML report goes to `htmlcov/`.

## 환경 변수 / Security & Configuration Tips
- `dot_env.example`을 복사해 `.env`를 만들고 값을 채웁니다.
- 새 Lambda 환경변수는 `.env`에 `LAMBDA_` prefix로 추가한 뒤 `just sync-env`로 동기화합니다.
- 별도 AWS 프로파일이 필요하면 `AWS_PROFILE=프로파일명 <command>` 형태로 실행합니다.
- 기본 배포 타깃: `AWS_PROFILE=sandbox`, `STACK_NAME=data-bolt`, `AWS_REGION=ap-northeast-2`.
- 로컬에서만 비밀값을 관리하고, 레포에 커밋하지 않습니다.

## Coding Style & Naming Conventions
- Python 3.13, source under `src/` (module name `data_bolt`).
- Formatting via Ruff (`line-length = 100`, double quotes). Run `uv run ruff format`.
- Linting via Ruff rules `B`, `I`, `RUF`, `UP` and `uv run ruff check --fix`.
- Type checking via strict mypy: `uv run mypy`.
- Use `snake_case` for functions/variables; tests named `test_*.py`.

## 테스트/품질
```bash
uv run ruff check .
uv run ruff format .
uv run mypy src/
uv run pytest
```

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio`, `pytest-cov`.
- Markers available: `unit`, `integration`, `slow` (e.g., `uv run pytest -m unit`).
- Prefer small, focused tests in `tests/` that mirror module names.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and can be English or Korean (no enforced prefix).
- PRs should include: summary, test results/commands, and any config or deployment impact.
- If Slack message formats change, include before/after examples in the PR.

## Agent-Specific Instructions
- When using a specific AWS CLI profile, always prefix commands with `AWS_PROFILE=<profile>`.
- Default deployment targets for this repo: `AWS_PROFILE=sandbox`, `STACK_NAME=data-bolt`, `AWS_REGION=ap-northeast-2`.

## 문서
- `docs/SLACK_GUIDE.md`: 메시지 포맷, 실패 대응, 트러블슈팅
