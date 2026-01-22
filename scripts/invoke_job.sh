#!/usr/bin/env bash
set -euo pipefail

AWS_CMD="${AWS_CMD:-aws}"
AWS_REGION="${AWS_REGION:-ap-northeast-2}"
STACK_NAME="${STACK_NAME:-sandbox}"
JOB_FUNCTION_NAME_FALLBACK="${JOB_FUNCTION_NAME:-JobFunction}"

job_function="$(
  AWS_CMD="$AWS_CMD" AWS_REGION="$AWS_REGION" STACK_NAME="$STACK_NAME" \
    OUTPUT_KEY="JobFunctionName" ALLOW_MISSING=1 \
    "$(dirname "$0")/get_stack_output.sh" || true
)"

if [[ -z "$job_function" ]]; then
  job_function="$JOB_FUNCTION_NAME_FALLBACK"
fi

echo "Invoking $job_function" >&2

$AWS_CMD lambda invoke \
  --region "$AWS_REGION" \
  --function-name "$job_function" \
  --payload fileb://event.json out.json >/dev/null

cat out.json
