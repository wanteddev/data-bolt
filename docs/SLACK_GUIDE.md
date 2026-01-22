# Slack 봇 가이드

이 문서는 Data Bolt Slack 봇의 메시지 포맷과 에러 대응 방법을 설명합니다.

## 목차

- [메시지 포맷 예시](#메시지-포맷-예시)
- [응답 패턴](#응답-패턴)
- [실패 대응 매뉴얼](#실패-대응-매뉴얼)
- [트러블슈팅](#트러블슈팅)
- [모니터링](#모니터링)

---

## 메시지 포맷 예시

### 기본 텍스트 메시지

```python
# 간단한 인사
say(f"안녕하세요 <@{user_id}>!")

# 멘션 포함
say(f"<@{user_id}>님의 요청을 처리했습니다.")
```

### 이모지 활용 패턴

| 상황 | 이모지 | 예시 |
|------|--------|------|
| 성공 | `:white_check_mark:` | `:white_check_mark: 작업이 완료되었습니다.` |
| 실패 | `:x:` | `:x: 오류가 발생했습니다.` |
| 처리 중 | `:hourglass_flowing_sand:` | `:hourglass_flowing_sand: 처리 중입니다...` |
| 경고 | `:warning:` | `:warning: 주의가 필요합니다.` |
| 정보 | `:information_source:` | `:information_source: 참고 사항입니다.` |

### response_type 옵션

슬래시 커맨드 응답 시 공개 범위를 설정할 수 있습니다:

```python
# 채널 전체에 공개 (모든 사용자가 볼 수 있음)
{
    "response_type": "in_channel",
    "text": ":white_check_mark: 작업이 완료되었습니다."
}

# 명령한 사용자에게만 표시 (비공개)
{
    "response_type": "ephemeral",
    "text": ":x: 권한이 없습니다."
}
```

### Block Kit 예시

복잡한 메시지는 Block Kit을 사용합니다:

```python
slack_client.chat_postMessage(
    channel=channel_id,
    blocks=[
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "작업 결과"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*상태:* 성공 :white_check_mark:\n*소요 시간:* 2.5초"
            }
        },
        {
            "type": "divider"
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "자세한 내용은 <https://example.com|여기>를 확인하세요."
            }
        }
    ],
    text="작업 결과"  # 알림용 fallback 텍스트
)
```

---

## 응답 패턴

### 1. 즉시 응답 (3초 이내)

Slack은 슬래시 커맨드에 **3초 이내** 응답을 요구합니다.

```python
@slack_app.command("/hello")
def handle_hello(ack: Ack, command: dict, say: Say) -> None:
    # 즉시 응답 (필수)
    ack()

    # 간단한 작업 후 메시지 전송
    say(f"안녕하세요 <@{command['user_id']}>!")
```

### 2. 지연 응답 (response_url)

3초를 초과하는 작업은 백그라운드로 처리합니다:

```python
@slack_app.command("/longtask")
def handle_long_task(ack: Ack, command: dict) -> None:
    # 즉시 임시 응답
    ack(":hourglass_flowing_sand: 처리 중입니다...")

    # 백그라운드 Lambda로 위임
    invoke_background(
        task_type="slash_command",
        payload={
            "command": "/longtask",
            "user_id": command["user_id"],
            "channel_id": command["channel_id"],
            "response_url": command["response_url"],  # 30분간 유효
        },
    )
```

백그라운드에서 response_url로 응답:

```python
async with httpx.AsyncClient() as client:
    await client.post(
        response_url,
        json={
            "response_type": "in_channel",
            "text": ":white_check_mark: 작업이 완료되었습니다!",
        },
    )
```

### 3. Web API 응답 (DM/채널)

이벤트 핸들러나 response_url이 없는 경우:

```python
slack_client.chat_postMessage(
    channel=channel_id,
    text="메시지 내용",
    thread_ts=thread_ts,  # 스레드 답글 (선택)
)
```

---

## 실패 대응 매뉴얼

### 에러 유형별 대응

| 에러 | 원인 | 대응 방법 |
|------|------|----------|
| `SlackBackgroundError` | 백그라운드 Lambda 호출 실패 | CloudWatch 로그 확인, IAM 권한 점검 |
| `SlackApiError` | Slack API 오류 | 토큰 유효성, rate limit, 권한 확인 |
| `httpx.HTTPError` | 네트워크 오류 | 재시도 로직 추가, 타임아웃 설정 |
| 3초 타임아웃 | 응답 지연 | 백그라운드 처리로 전환 |
| `BoltUnhandledRequestError` | 핸들러 미등록 | 해당 이벤트/커맨드 핸들러 추가 |

### 에러별 상세 대응

#### SlackBackgroundError

```python
try:
    invoke_background(task_type="my_task", payload={...})
except SlackBackgroundError as e:
    logger.error(f"백그라운드 작업 실패: {e}")
    say(":x: 요청 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
```

**확인 사항:**
1. `SLACK_BG_FUNCTION_NAME` 환경변수 설정 확인
2. Web Lambda → SlackBg Lambda 호출 권한 확인
3. SlackBg Lambda CloudWatch 로그 확인

#### SlackApiError

```python
from slack_sdk.errors import SlackApiError

try:
    slack_client.chat_postMessage(channel=channel_id, text="...")
except SlackApiError as e:
    error_code = e.response.get("error", "unknown")
    logger.error(f"Slack API 오류: {error_code}")

    if error_code == "channel_not_found":
        # 채널 접근 권한 확인
        pass
    elif error_code == "ratelimited":
        # 재시도 대기
        retry_after = int(e.response.headers.get("Retry-After", 1))
        await asyncio.sleep(retry_after)
```

**주요 에러 코드:**
| 코드 | 의미 | 해결 방법 |
|------|------|----------|
| `invalid_auth` | 토큰 무효 | SLACK_BOT_TOKEN 재발급 |
| `channel_not_found` | 채널 없음/권한 없음 | 봇을 채널에 초대 |
| `ratelimited` | API 호출 제한 | Retry-After 헤더만큼 대기 |
| `missing_scope` | 권한 부족 | OAuth 스코프 추가 |

---

## 트러블슈팅

### 체크리스트

#### 1. 환경변수 확인

```bash
# CloudFormation 파라미터 확인
aws cloudformation describe-stacks \
  --stack-name sandbox \
  --query "Stacks[0].Parameters"
```

필수 환경변수:
- `SLACK_BOT_TOKEN`: `xoxb-`로 시작
- `SLACK_SIGNING_SECRET`: Slack 앱 Basic Information에서 확인
- `SLACK_BG_FUNCTION_NAME`: 자동 설정됨

#### 2. Lambda 권한 확인

```bash
# Web Lambda 역할 정책 확인
aws iam list-attached-role-policies \
  --role-name sandbox-web-role

# Slack BG Lambda 호출 권한 확인 (AWSLambda_FullAccess 포함되어야 함)
```

#### 3. Slack 앱 설정 확인

1. **Event Subscriptions**
   - Request URL: `{Function URL}slack/events`
   - URL이 "Verified" 상태인지 확인

2. **OAuth & Permissions**
   - Bot Token Scopes 확인
   - 필수: `chat:write`, `commands`

3. **Slash Commands**
   - Request URL이 올바른지 확인

#### 4. 로그 확인

```bash
# Web Lambda 로그
aws logs tail /aws/lambda/sandbox-WebFunction --follow

# Slack Background Lambda 로그
aws logs tail /aws/lambda/sandbox-SlackBgFunction --follow
```

### 자주 발생하는 문제

#### "dispatch_failed" 에러

**원인**: Slack이 3초 내 응답을 받지 못함

**해결**:
```python
@slack_app.command("/mycommand")
def handle(ack: Ack, command: dict) -> None:
    ack()  # 반드시 먼저 호출!
    # 이후 작업...
```

#### 메시지가 전송되지 않음

**확인 사항**:
1. 봇이 해당 채널에 초대되어 있는지
2. `chat:write` 스코프가 있는지
3. channel_id가 올바른지 (C로 시작: 공개채널, D로 시작: DM)

#### 백그라운드 작업이 실행되지 않음

**확인 사항**:
1. SlackBgFunction이 배포되어 있는지
2. Web Lambda가 SlackBg Lambda를 호출할 권한이 있는지
3. 페이로드 형식이 올바른지

---

## 모니터링

### CloudWatch 로그 필터

에러 로그 필터링:

```bash
# 에러 로그만 조회
aws logs filter-log-events \
  --log-group-name /aws/lambda/sandbox-WebFunction \
  --filter-pattern "ERROR"

# Slack API 에러만 조회
aws logs filter-log-events \
  --log-group-name /aws/lambda/sandbox-SlackBgFunction \
  --filter-pattern "SlackApiError"
```

### CloudWatch 알람 설정 예시

Lambda 에러 알람:

```yaml
# CloudFormation 예시
ErrorAlarm:
  Type: AWS::CloudWatch::Alarm
  Properties:
    AlarmName: sandbox-errors
    MetricName: Errors
    Namespace: AWS/Lambda
    Statistic: Sum
    Period: 300
    EvaluationPeriods: 1
    Threshold: 1
    ComparisonOperator: GreaterThanOrEqualToThreshold
    Dimensions:
      - Name: FunctionName
        Value: !Ref WebFunction
```

### 유용한 메트릭

| 메트릭 | 설명 | 권장 임계값 |
|--------|------|------------|
| `Errors` | Lambda 에러 수 | > 0 알람 |
| `Duration` | 실행 시간 | p99 < 3초 (Web) |
| `Throttles` | 스로틀링 횟수 | > 0 알람 |
| `ConcurrentExecutions` | 동시 실행 수 | 모니터링 |
