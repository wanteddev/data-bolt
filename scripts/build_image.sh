#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
  TAG="${IMAGE_TAG:-}"
fi
if [[ -z "$TAG" ]]; then
  TAG="v$(date -u +%Y%m%d%H%M%S)"
fi
if [[ -z "${IMAGE_NAME:-}" ]]; then
  echo "IMAGE_NAME environment variable is not set." >&2
  exit 1
fi

FULL="${IMAGE_NAME}:${TAG}"
CONTEXT="${BUILD_CONTEXT_PATH:-.}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-Dockerfile}"
REGISTRY_HOST="${IMAGE_REGISTRY:-${IMAGE_NAME%/*}}"
CACHE_TAG="${IMAGE_CACHE_TAG:-cache}"
CACHE_REF="${IMAGE_CACHE_REF:-}"
USE_ECR_CACHE="${USE_ECR_CACHE:-1}"
AWS_CMD="${AWS_CMD:-aws}"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  DIRTY_COUNT="$(git status --porcelain | wc -l | tr -d ' ')"
  if [[ "${DIRTY_COUNT:-0}" -ne 0 ]]; then
    echo "Uncommitted changes detected (${DIRTY_COUNT} files). Aborting build." >&2
    git status --porcelain >&2
    exit 1
  fi
fi

if [[ -z "$CACHE_REF" ]]; then
  if [[ -f "${IMAGE_TAG_CACHE:-.last_image_tag}" ]]; then
    LAST_TAG="$(<"${IMAGE_TAG_CACHE:-.last_image_tag}")"
    CACHE_REF="${IMAGE_NAME}:${LAST_TAG}"
  else
    CACHE_REF="${IMAGE_NAME}:${CACHE_TAG}"
  fi
fi

cache_args=()
if [[ "$USE_ECR_CACHE" == "1" ]]; then
  if [[ "$REGISTRY_HOST" == *.ecr.*.amazonaws.com ]]; then
    REGION="${AWS_REGION:-}"
    if [[ -z "$REGION" ]]; then
      REGION="$(awk -F. '{for (i=1;i<=NF;i++) if ($i=="ecr") {print $(i+1); exit}}' <<<"$REGISTRY_HOST")"
    fi
    if command -v "$AWS_CMD" >/dev/null 2>&1 && [[ -n "$REGION" ]]; then
      "$AWS_CMD" ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY_HOST" >/dev/null || true
    fi
  fi

  if docker pull "$CACHE_REF" >/dev/null 2>&1; then
    cache_args+=(--cache-from "$CACHE_REF")
    echo "Using build cache from $CACHE_REF"
  else
    echo "No cache image found at $CACHE_REF (build will proceed without remote cache)"
  fi
fi

DOCKER_BUILDKIT=1 docker build \
  --platform linux/arm64 \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  "${cache_args[@]}" \
  -t "$FULL" \
  -f "$DOCKERFILE_PATH" \
  "$CONTEXT"
printf '%s\n' "$TAG" > "${IMAGE_TAG_CACHE:-.last_image_tag}"
if command -v python3 >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_SHA="$(git rev-parse HEAD)"
  DIRTY_COUNT="$(git status --porcelain | wc -l | tr -d ' ')"
  BUILT_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 - <<PY > "${IMAGE_TAG_CACHE_META:-.last_image_tag.json}"
import json
print(json.dumps({
  "tag": "${TAG}",
  "git_sha": "${GIT_SHA}",
  "dirty_count": int("${DIRTY_COUNT}" or 0),
  "built_at": "${BUILT_AT}"
}, ensure_ascii=False))
PY
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_SHA="$(git rev-parse HEAD)"
  if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null 2>&1; then
    EXISTING_SHA="$(git rev-list -n 1 "${TAG}")"
    if [[ "$EXISTING_SHA" != "$GIT_SHA" ]]; then
      echo "Git tag '${TAG}' already exists but points to different commit (${EXISTING_SHA}). Build tag conflict." >&2
      exit 1
    fi
  else
    git tag "${TAG}" "${GIT_SHA}"
  fi
fi

echo "Built image $FULL"
