#!/usr/bin/env bash
set -euo pipefail

AWS_CMD="${AWS_CMD:-aws}"
AWS_REGION="${AWS_REGION:-ap-northeast-2}"
STACK_NAME="${STACK_NAME:-sandbox}"
OUTPUT_KEY="${OUTPUT_KEY:?OUTPUT_KEY must be set}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"

raw="$($AWS_CMD cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" --output json)"
value="$(RAW_JSON="$raw" OUTPUT_KEY="$OUTPUT_KEY" python3 -c 'import json, os
raw = os.environ.get("RAW_JSON", "{}")
key = os.environ.get("OUTPUT_KEY", "")
doc = json.loads(raw)
stacks = doc.get("Stacks") or []
stack = stacks[0] if stacks else {}
for out in (stack.get("Outputs") or []):
    if out.get("OutputKey") == key:
        v = out.get("OutputValue")
        print(v if isinstance(v, str) else "")
        raise SystemExit(0)
print("")
')"

if [[ -z "$value" || "$value" == "None" || "$value" == "null" ]]; then
  if [[ "$ALLOW_MISSING" == "1" ]]; then
    exit 0
  fi
  echo "Output not found: stack=${STACK_NAME} key=${OUTPUT_KEY}" >&2
  exit 1
fi

echo "$value"
