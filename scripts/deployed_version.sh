#!/usr/bin/env bash
set -euo pipefail

AWS_CMD="${AWS_CMD:-aws}"
AWS_REGION="${AWS_REGION:-ap-northeast-2}"
STACK_NAME="${STACK_NAME:-sandbox}"
IMAGE_REPO="${IMAGE_REPO:-sandbox}"

image_uri="$($AWS_CMD cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --query "Stacks[0].Parameters[?ParameterKey=='ImageUri'].ParameterValue | [0]" \
  --output text)"

if [[ "$image_uri" == "None" || "$image_uri" == "null" || -z "$image_uri" ]]; then
  echo "ImageUri not found for stack ${STACK_NAME}" >&2
  exit 1
fi

echo "stack=${STACK_NAME}"
echo "region=${AWS_REGION}"
echo "aws_profile=${AWS_PROFILE:-}"
echo "image_uri=${image_uri}"

digest=""
case "$image_uri" in
  *@sha256:*)
    digest="${image_uri##*@}"
    ;;
esac

if [[ -z "$digest" ]]; then
  exit 0
fi

tags_json="$($AWS_CMD ecr describe-images \
  --repository-name "$IMAGE_REPO" \
  --region "$AWS_REGION" \
  --image-ids "imageDigest=${digest}" \
  --query "imageDetails[0].imageTags" \
  --output json)"
echo "ecr_tags=${tags_json}"

picked_tag="$(TAGS_JSON="$tags_json" python3 -c 'import json, os, re
tags = json.loads(os.environ.get("TAGS_JSON", "[]") or "[]")
print(next((t for t in tags if isinstance(t, str) and re.fullmatch(r"v\\d{14}", t)), ""))
')"

if [[ -z "$picked_tag" ]]; then
  exit 0
fi

echo "git_tag=${picked_tag}"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  && git rev-parse -q --verify "refs/tags/${picked_tag}" >/dev/null 2>&1; then
  echo "git_sha=$(git rev-list -n 1 "${picked_tag}")"
else
  echo "git_sha=(tag not found locally)"
fi
