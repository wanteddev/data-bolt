# Data Bolt

A Slack bot on AWS Lambda with Litestar

## 아키텍처

[Litestar](https://litestar.dev/)로 구축되고 [Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter)를 사용하여 AWS Lambda에 배포되는 서버리스 애플리케이션입니다.

**구성 요소:**
- **Web Function**: Function URL을 통해 HTTP 요청 처리

- **Slack Background Function**: 장시간 Slack 작업을 비동기로 처리
- **Single Container Image**: 모든 함수가 동일한 Docker 이미지 공유

## 사전 요구사항

- [uv](https://docs.astral.sh/uv/) - Python 패키지 관리자
- [just](https://just.systems/) - 커맨드 러너
- [Docker](https://www.docker.com/) - 컨테이너 이미지 빌드용
- 적절한 자격 증명이 구성된 AWS CLI

## 빠른 시작

### 1. 의존성 설치

```bash
uv sync --frozen
```

### 2. 로컬 실행

```bash
just serve
```

http://localhost:8080 에서 앱에 접근할 수 있습니다.

### 2-1. Slack 없이 로컬 시뮬레이션 (`botctl`)

`botctl`은 Slack 이벤트 없이 메시지 처리 흐름을 바로 테스트하는 CLI입니다.

```bash
uv sync --frozen
uv run botctl sync-laas-key --profile sandbox --region ap-northeast-2
uv run botctl simulate --text "안녕하세요"
uv run botctl simulate --case sql_gen --json
uv run botctl chat
uv run botctl chat --thread-ts demo-thread
```

`sync-laas-key`는 SSM 파라미터(`/DATA/PIPELINE/API_KEY/OPENAI`)를 읽어
`.env`의 `LAAS_API_KEY`로 저장합니다. 이후 `simulate`는 기본적으로 `.env`를 읽어 사용합니다.

다턴 컨텍스트 정책:
- `botctl chat`: 단일 프로세스에서 다턴 대화를 이어갈 수 있습니다.
- `LANGGRAPH_CHECKPOINT_BACKEND=memory`: 프로세스 종료 시 컨텍스트가 사라집니다.
- `LANGGRAPH_CHECKPOINT_BACKEND=postgres`: 동일 `--thread-ts`로 프로세스 간 대화 재개가 가능합니다.
- `LANGGRAPH_CHECKPOINT_BACKEND=dynamodb`: 동일 `--thread-ts`로 Lambda 간 대화 재개가 가능합니다.
- `dynamodb` backend에서 체크포인트 저장소 오류가 발생하면 fallback 없이 즉시 실패합니다 (fail-fast).
- `botctl simulate --thread-ts ...`는 memory backend에서 거부됩니다.

LLM 라우팅/플래너 정책:
- `BIGQUERY_INTENT_LLM_ENABLED=true`: 의도 분류를 LLM으로 수행
- `BIGQUERY_INTENT_CONFIDENCE_THRESHOLD=0.55`: 낮은 신뢰도는 자유 대화 경로로 보수 라우팅
- `BIGQUERY_CHAT_PLANNER_MAX_ACTIONS=2`: 자유 대화 중 호출 가능한 데이터 액션 개수 제한
- `BIGQUERY_CHAT_ALLOW_EXECUTE_IN_CHAT=false`: 자유 대화 경로에서 execute 액션 기본 차단

### 3. 빌드 및 배포

```bash
# (최초 1회 또는 값 변경 시) Slack 비밀값을 SSM에 동기화
AWS_PROFILE=sandbox AWS_REGION=ap-northeast-2 just sync-secrets

# 컨테이너 이미지 빌드
just build

# AWS에 배포
just deploy

# 배포된 URL 확인
just url
```

이미지 매니페스트 정책:
- `just build`는 Lambda 호환성을 위해 `buildx --provenance=false --sbom=false --load`를 강제합니다.
- `just deploy`는 ECR에 푸시된 이미지의 `imageManifestMediaType`을 검사하고, Lambda 비호환 포맷(OCI index / Docker manifest list)이면 배포를 중단합니다.

## 사용 가능한 명령어

| 명령어 | 설명 |
|--------|------|
| `just serve` | 로컬 개발 서버 실행 |
| `just build [tag]` | 컨테이너 이미지 빌드 |
| `just deploy [tag]` | ECR에 푸시하고 스택 배포 |
| `just url` | 배포된 Function URL 확인 |
| `just invoke-job` | 예약된 작업 수동 실행 |
| `just deployed-version` | 배포된 이미지 정보 표시 |
| `just lock` | 의존성 lockfile 업데이트 |
| `just sync-env` | `.env`의 LAMBDA_* 변수를 CloudFormation에 동기화 |
| `just sync-secrets [env_file]` | Slack 비밀값을 SSM SecureString에 동기화 |

## 프로젝트 구조

```
data_bolt/
├── src/data_bolt/
│   ├── __init__.py
│   ├── app.py              # Litestar 애플리케이션

│   └── slack/
│       ├── __init__.py
│       ├── app.py          # Slack Bolt 앱 설정
│       ├── handlers.py     # 커맨드 & 이벤트 핸들러
│       └── background.py   # 백그라운드 작업 프로세서
├── scripts/
│   ├── sync_env.py         # 환경변수 동기화 스크립트
│   └── ...                 # 빌드/배포 스크립트
├── dot_env.example         # 환경변수 예시 파일 (cp dot_env.example .env)
├── docs/
│   └── SLACK_GUIDE.md      # Slack 알림 포맷 및 실패 대응 가이드
├── template.yaml           # CloudFormation 스택
├── Dockerfile
├── pyproject.toml
└── justfile
```

## 문서

- [Slack 봇 가이드](docs/SLACK_GUIDE.md) - 메시지 포맷 예시, 실패 대응 매뉴얼, 트러블슈팅

## 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/` | GET | 헬스 체크 |
| `/healthz` | GET | 로드 밸런서 헬스 체크 |

| `/slack/events` | POST | Slack 웹훅 엔드포인트 (커맨드, 이벤트, 인터랙션) |
| `/slack/background` | POST | 백그라운드 작업 프로세서 (내부 사용) |

## 설정

### 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `AWS_PROFILE` | AWS CLI 프로필 | `default` |
| `AWS_REGION` | AWS 리전 | `ap-northeast-2` |
| `STACK_NAME` | CloudFormation 스택 이름 | `data_bolt` |

### CloudFormation 파라미터

| 파라미터 | 설명 |
|----------|------|
| `ImageUri` | ECR 이미지 참조 |
| `SlackBotToken` | Slack Bot User OAuth Token (deploy 시 SSM에서 조회) |
| `SlackSigningSecret` | Slack Signing Secret (deploy 시 SSM에서 조회) |

### Lambda 환경변수 추가하기

앱 개발 중 새로운 환경변수(예: API 키, 데이터베이스 URL)를 추가하려면:

**1. `.env` 파일에 `LAMBDA_` prefix로 변수 추가**

```bash
# .env
LAMBDA_DATABASE_URL=postgresql://user:pass@host:5432/db
LAMBDA_OPENAI_API_KEY=sk-...
LAMBDA_FEATURE_FLAG=true
```

**2. 동기화 실행**

```bash
just sync-env
```

이 명령은 자동으로:
- `template.yaml`에 CloudFormation 파라미터 추가 (예: `DatabaseUrl`)
- Lambda 함수들의 환경변수에 참조 추가 (예: `DATABASE_URL: !Ref DatabaseUrl`)
- `deploy_stack.sh`에 파라미터 전달 로직 추가

**3. 배포**

```bash
just build && just deploy
```

**변환 규칙:**
| .env 변수 | CFN 파라미터 | Lambda 환경변수 |
|-----------|--------------|-----------------|
| `LAMBDA_DATABASE_URL` | `DatabaseUrl` | `DATABASE_URL` |
| `LAMBDA_OPENAI_API_KEY` | `OpenaiApiKey` | `OPENAI_API_KEY` |



## Slack 봇 설정

### 1. Slack 앱 생성

1. [Slack API Apps](https://api.slack.com/apps)로 이동
2. **Create New App** → **From scratch** 클릭
3. 앱 이름 입력 및 워크스페이스 선택

### 2. 봇 권한 설정

**OAuth & Permissions**로 이동하여 Bot Token Scopes 추가:

| Scope | 설명 |
|-------|------|
| `chat:write` | 메시지 전송 |
| `commands` | 슬래시 커맨드 추가 |
| `app_mentions:read` | @멘션 수신 |
| `im:history` | DM 메시지 읽기 |
| `im:write` | DM 전송 |

### 3. 이벤트 활성화

1. **Event Subscriptions**로 이동
2. **Enable Events**를 On으로 토글
3. **Request URL** 설정: `{Function URL}slack/events`
4. 봇 이벤트 구독:
   - `app_mention`
   - `message.im`

### 4. 슬래시 커맨드 추가

**Slash Commands** → **Create New Command**로 이동:

| 커맨드 | Request URL | 설명 |
|--------|-------------|------|
| `/hello` | `{Function URL}slack/events` | 인사 커맨드 예시 |
| `/longtask` | `{Function URL}slack/events` | 백그라운드 작업 예시 |

### 5. 워크스페이스에 설치

1. **Install App**으로 이동
2. **Install to Workspace** 클릭
3. **Bot User OAuth Token** 복사 (`xoxb-`로 시작)

### 6. Signing Secret 확인

1. **Basic Information**으로 이동
2. **Signing Secret** 복사

### 7. Slack 자격 증명으로 배포

Slack 자격 증명은 SSM SecureString으로 저장 후 배포합니다:

```bash
# 1) 로컬 .env 값 -> SSM 동기화
AWS_PROFILE=sandbox AWS_REGION=ap-northeast-2 just sync-secrets

# 2) 배포
AWS_PROFILE=sandbox AWS_REGION=ap-northeast-2 just deploy
```

SSM 경로를 바꾸고 싶다면:

```bash
AWS_PROFILE=sandbox AWS_REGION=ap-northeast-2 \
SLACK_BOT_TOKEN_SSM_PATH=/data-bolt/slack/bot-token \
SLACK_SIGNING_SECRET_SSM_PATH=/data-bolt/slack/signing-secret \
just deploy
```

## Slack 핸들러 추가하기

핸들러 예시와 백그라운드 패턴은 `docs/SLACK_GUIDE.md`에 정리했습니다.

## 라이선스

MIT
