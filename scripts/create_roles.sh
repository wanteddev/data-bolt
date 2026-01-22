#!/usr/bin/env bash
set -euo pipefail

# IAM 역할 생성 스크립트
# 이 스크립트는 CloudFormation 스택 배포 전에 필요한 IAM 역할들을 생성합니다.

REGION="${AWS_REGION:-ap-northeast-2}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-216989105690}"
STACK_NAME="${STACK_NAME:-sandbox}"
AWS_CMD="${AWS_CMD:-aws}"

echo "Creating IAM roles for stack: $STACK_NAME"
echo "Region: $REGION, Account: $ACCOUNT_ID"

# Lambda 신뢰 정책
LAMBDA_TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "lambda.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}'

# Scheduler 신뢰 정책
SCHEDULER_TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "scheduler.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}'

create_role_if_not_exists() {
  local role_name="$1"
  local trust_policy="$2"
  local description="$3"

  if $AWS_CMD iam get-role --role-name "$role_name" --region "$REGION" >/dev/null 2>&1; then
    echo "Role $role_name already exists, skipping..."
  else
    echo "Creating role: $role_name"
    $AWS_CMD iam create-role \
      --role-name "$role_name" \
      --assume-role-policy-document "$trust_policy" \
      --description "$description" \
      --region "$REGION"
  fi
}

attach_policy_if_not_attached() {
  local role_name="$1"
  local policy_arn="$2"

  if $AWS_CMD iam list-attached-role-policies --role-name "$role_name" --region "$REGION" \
    | grep -q "$policy_arn"; then
    echo "Policy $policy_arn already attached to $role_name, skipping..."
  else
    echo "Attaching policy $policy_arn to $role_name"
    $AWS_CMD iam attach-role-policy \
      --role-name "$role_name" \
      --policy-arn "$policy_arn" \
      --region "$REGION"
  fi
}

# 1. Web 함수 역할
WEB_ROLE_NAME="${STACK_NAME}-web-role"
create_role_if_not_exists "$WEB_ROLE_NAME" "$LAMBDA_TRUST_POLICY" "Lambda execution role for $STACK_NAME web function"
attach_policy_if_not_attached "$WEB_ROLE_NAME" "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
# Web 역할에 Lambda 호출 권한 추가 (Slack 백그라운드 함수 호출용)
attach_policy_if_not_attached "$WEB_ROLE_NAME" "arn:aws:iam::aws:policy/AWSLambda_FullAccess"


# Slack 백그라운드 함수 역할
SLACK_BG_ROLE_NAME="${STACK_NAME}-slack-bg-role"
create_role_if_not_exists "$SLACK_BG_ROLE_NAME" "$LAMBDA_TRUST_POLICY" "Lambda execution role for $STACK_NAME Slack background function"
attach_policy_if_not_attached "$SLACK_BG_ROLE_NAME" "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"

echo ""
echo "IAM roles created successfully!"
echo ""
echo "Created roles:"
echo "  - $WEB_ROLE_NAME"

echo "  - $SLACK_BG_ROLE_NAME"
