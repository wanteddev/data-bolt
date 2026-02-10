#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.env}"
REGION="${AWS_REGION:?AWS_REGION must be set}"
AWS_CMD="${AWS_CMD:-aws}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
fi

SLACK_BOT_TOKEN_SSM_PATH="${SLACK_BOT_TOKEN_SSM_PATH:-/data-bolt/slack/bot-token}"
SLACK_SIGNING_SECRET_SSM_PATH="${SLACK_SIGNING_SECRET_SSM_PATH:-/data-bolt/slack/signing-secret}"

if [[ -z "${SLACK_BOT_TOKEN:-}" ]]; then
  echo "SLACK_BOT_TOKEN is not set. Set it in ${ENV_FILE} or environment." >&2
  exit 1
fi

if [[ -z "${SLACK_SIGNING_SECRET:-}" ]]; then
  echo "SLACK_SIGNING_SECRET is not set. Set it in ${ENV_FILE} or environment." >&2
  exit 1
fi

$AWS_CMD ssm put-parameter \
  --name "$SLACK_BOT_TOKEN_SSM_PATH" \
  --type SecureString \
  --value "$SLACK_BOT_TOKEN" \
  --overwrite \
  --region "$REGION" >/dev/null
echo "Updated SSM SecureString: $SLACK_BOT_TOKEN_SSM_PATH"

$AWS_CMD ssm put-parameter \
  --name "$SLACK_SIGNING_SECRET_SSM_PATH" \
  --type SecureString \
  --value "$SLACK_SIGNING_SECRET" \
  --overwrite \
  --region "$REGION" >/dev/null
echo "Updated SSM SecureString: $SLACK_SIGNING_SECRET_SSM_PATH"
