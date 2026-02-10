#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
  TAG="${IMAGE_TAG:-}"
  [[ -n "$TAG" ]] && TAG_SOURCE="env"
else
  TAG_SOURCE="arg"
fi
if [[ -z "$TAG" ]] && [[ -f "${IMAGE_TAG_CACHE:-.last_image_tag}" ]]; then
  TAG="$(<"${IMAGE_TAG_CACHE:-.last_image_tag}")"
  TAG_SOURCE="cache"
fi
if [[ -z "$TAG" ]]; then
  echo "Cannot determine image tag. Provide as argument or set IMAGE_TAG environment variable." >&2
  exit 1
fi

if [[ -z "${IMAGE_NAME:-}" ]]; then
  echo "IMAGE_NAME environment variable is not set." >&2
  exit 1
fi

FULL="${IMAGE_NAME}:${TAG}"
REGION="${AWS_REGION:?AWS_REGION must be set}"
STACK="${STACK_NAME:?STACK_NAME must be set}"
TEMPLATE="${TEMPLATE_FILE:-template.yaml}"
REPO="${IMAGE_REPO:-${IMAGE_NAME##*/}}"
REGISTRY_HOST="${IMAGE_REGISTRY:-${IMAGE_NAME%/*}}"

if [[ "${TAG_SOURCE:-}" == "cache" ]] && [[ -z "${FORCE_DEPLOY:-}" ]]; then
  meta_file="${IMAGE_TAG_CACHE_META:-.last_image_tag.json}"
  if [[ ! -f "$meta_file" ]]; then
    echo "Build meta file (${meta_file}) not found. Run 'just build' first." >&2
    echo "To force deploy, use FORCE_DEPLOY=1 just deploy" >&2
    exit 1
  fi
  if command -v python3 >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    read -r BUILT_SHA BUILT_DIRTY BUILT_TAG < <(
      python3 - <<PY
import json
with open("${meta_file}", "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.get("git_sha", ""), data.get("dirty_count", 0), data.get("tag", ""))
PY
    )
    CURRENT_SHA="$(git rev-parse HEAD)"
    CURRENT_DIRTY="$(git status --porcelain | wc -l | tr -d ' ')"
    if [[ -n "$BUILT_SHA" && "$BUILT_SHA" != "$CURRENT_SHA" ]]; then
      echo "Latest build tag (${BUILT_TAG}) is based on $BUILT_SHA. Current HEAD=$CURRENT_SHA, rebuild required." >&2
      echo "To force deploy, use FORCE_DEPLOY=1 just deploy" >&2
      exit 1
    fi
    if [[ "$CURRENT_DIRTY" -ne 0 && "$BUILT_DIRTY" -eq 0 ]]; then
      echo "Working directory has uncommitted changes. Latest changes may not be in the image." >&2
      echo "To force deploy, use FORCE_DEPLOY=1 just deploy" >&2
      exit 1
    fi
  fi
fi

if ! ${AWS_CMD:-aws} ecr describe-repositories --repository-name "$REPO" --region "$REGION" >/dev/null 2>&1; then
  ${AWS_CMD:-aws} ecr create-repository --repository-name "$REPO" --image-scanning-configuration scanOnPush=true --region "$REGION"
fi

${AWS_CMD:-aws} ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY_HOST"
docker push "$FULL"

manifest_media_type=$(${AWS_CMD:-aws} ecr describe-images --repository-name "$REPO" --image-ids imageTag="$TAG" --region "$REGION" --query "imageDetails[0].imageManifestMediaType" --output text)
if [[ "$manifest_media_type" == "application/vnd.oci.image.index.v1+json" || "$manifest_media_type" == "application/vnd.docker.distribution.manifest.list.v2+json" ]]; then
  echo "Unsupported image manifest media type for Lambda: $manifest_media_type" >&2
  echo "Run 'just build' with the updated build script (buildx --provenance=false --sbom=false --load)." >&2
  exit 1
fi

digest=$(${AWS_CMD:-aws} ecr describe-images --repository-name "$REPO" --image-ids imageTag="$TAG" --region "$REGION" --query "imageDetails[0].imageDigest" --output text)
if [[ -z "$digest" || "$digest" == "None" || "$digest" == "null" ]]; then
  echo "Failed to fetch image digest for tag $TAG" >&2
  exit 1
fi

image_uri="${IMAGE_NAME}@${digest}"
echo "Deploying $image_uri"

PARAM_OVERRIDES="ImageUri=$image_uri"

SLACK_BOT_TOKEN_SSM_PATH="${SLACK_BOT_TOKEN_SSM_PATH:-/data-bolt/slack/bot-token}"
SLACK_SIGNING_SECRET_SSM_PATH="${SLACK_SIGNING_SECRET_SSM_PATH:-/data-bolt/slack/signing-secret}"

if [[ -n "${SLACK_BOT_TOKEN:-}" ]]; then
  ${AWS_CMD:-aws} ssm put-parameter \
    --name "$SLACK_BOT_TOKEN_SSM_PATH" \
    --type SecureString \
    --value "$SLACK_BOT_TOKEN" \
    --overwrite \
    --region "$REGION" >/dev/null
  echo "Updated SSM SecureString: $SLACK_BOT_TOKEN_SSM_PATH"
fi

if [[ -n "${SLACK_SIGNING_SECRET:-}" ]]; then
  ${AWS_CMD:-aws} ssm put-parameter \
    --name "$SLACK_SIGNING_SECRET_SSM_PATH" \
    --type SecureString \
    --value "$SLACK_SIGNING_SECRET" \
    --overwrite \
    --region "$REGION" >/dev/null
  echo "Updated SSM SecureString: $SLACK_SIGNING_SECRET_SSM_PATH"
fi

resolved_slack_bot_token=$(${AWS_CMD:-aws} ssm get-parameter --name "$SLACK_BOT_TOKEN_SSM_PATH" --with-decryption --region "$REGION" --query "Parameter.Value" --output text)
resolved_slack_signing_secret=$(${AWS_CMD:-aws} ssm get-parameter --name "$SLACK_SIGNING_SECRET_SSM_PATH" --with-decryption --region "$REGION" --query "Parameter.Value" --output text)

if [[ -z "$resolved_slack_bot_token" || "$resolved_slack_bot_token" == "None" ]]; then
  echo "Failed to resolve Slack bot token from SSM path: $SLACK_BOT_TOKEN_SSM_PATH" >&2
  exit 1
fi
if [[ -z "$resolved_slack_signing_secret" || "$resolved_slack_signing_secret" == "None" ]]; then
  echo "Failed to resolve Slack signing secret from SSM path: $SLACK_SIGNING_SECRET_SSM_PATH" >&2
  exit 1
fi

PARAM_OVERRIDES="$PARAM_OVERRIDES SlackBotToken=$resolved_slack_bot_token"
PARAM_OVERRIDES="$PARAM_OVERRIDES SlackSigningSecret=$resolved_slack_signing_secret"
[[ -n "${LANGGRAPH_CHECKPOINT_BACKEND:-}" ]] && PARAM_OVERRIDES="$PARAM_OVERRIDES LanggraphCheckpointBackend=$LANGGRAPH_CHECKPOINT_BACKEND"

${AWS_CMD:-aws} cloudformation deploy \
  --stack-name "$STACK" \
  --template-file "$TEMPLATE" \
  --region "$REGION" \
  --capabilities CAPABILITY_IAM \
  --no-fail-on-empty-changeset \
  --parameter-overrides $PARAM_OVERRIDES

printf '%s\n' "$TAG" > "${IMAGE_TAG_CACHE:-.last_image_tag}"
