set shell := ["bash", "-lc"]

# Default values (override with environment variables)
image_registry := env('IMAGE_REGISTRY', '216989105690.dkr.ecr.ap-northeast-2.amazonaws.com')
image_repo := env('IMAGE_REPO', 'sandbox')
image_name := env('IMAGE_NAME', image_registry + "/" + image_repo)
uv := env('UV', 'uv')
aws := env('AWS', 'aws')
aws_profile := env('AWS_PROFILE', 'default')
aws_region := env('AWS_REGION', 'ap-northeast-2')
stack_name := env('STACK_NAME', 'sandbox')

tag_cache := env('IMAGE_TAG_CACHE', '.last_image_tag')
dockerfile := env('DOCKERFILE_PATH', 'Dockerfile')
build_context := env('BUILD_CONTEXT_PATH', '.')

[doc("Show available recipes")]
default:
  @just --list --unsorted --justfile {{justfile()}}

[doc("Update dependency lockfile with uv")]
lock:
  {{uv}} lock
  cp uv.lock requirements.lock

[doc("Build container image (default tag: vYYYYMMDDHHMMSS)")]
build version="":
  IMAGE_REGISTRY={{image_registry}} IMAGE_REPO={{image_repo}} IMAGE_NAME={{image_name}} IMAGE_TAG={{version}} IMAGE_TAG_CACHE={{tag_cache}} DOCKERFILE_PATH={{dockerfile}} BUILD_CONTEXT_PATH={{build_context}} scripts/build_image.sh {{version}}

[doc("Push to ECR and deploy CloudFormation stack (default tag: latest build)")]
deploy version="":
  AWS_PROFILE={{aws_profile}} IMAGE_REGISTRY={{image_registry}} IMAGE_REPO={{image_repo}} IMAGE_NAME={{image_name}} IMAGE_TAG_CACHE={{tag_cache}} AWS_REGION={{aws_region}} STACK_NAME={{stack_name}} TEMPLATE_FILE=template.yaml AWS_CMD={{aws}} SLACK_BOT_TOKEN="$SLACK_BOT_TOKEN" SLACK_SIGNING_SECRET="$SLACK_SIGNING_SECRET" scripts/deploy_stack.sh {{version}}

[doc("Get deployed Function URL")]
url:
  @AWS_PROFILE={{aws_profile}} AWS_CMD={{aws}} AWS_REGION={{aws_region}} STACK_NAME={{stack_name}} OUTPUT_KEY=WebFunctionUrl scripts/get_stack_output.sh


[doc("Show currently deployed image/tag/commit info")]
deployed-version:
  @AWS_PROFILE={{aws_profile}} AWS_CMD={{aws}} AWS_REGION={{aws_region}} STACK_NAME={{stack_name}} IMAGE_REPO={{image_repo}} scripts/deployed_version.sh

[doc("Run Litestar dev server locally")]
serve:
  {{uv}} run litestar --app data_bolt.app:app run --host 0.0.0.0 --port 8080

[doc("Run LangGraph local dev server (loop + legacy graphs)")]
langgraph-dev port="8123":
  {{uv}} run langgraph dev --config langgraph.json --no-browser --port {{port}}

[doc("Sync LAMBDA_* env vars from .env to template.yaml and deploy_stack.sh")]
sync-env:
  {{uv}} run python scripts/sync_env.py

[doc("Sync Slack secrets from .env to SSM SecureString parameters")]
sync-secrets env_file=".env":
  AWS_PROFILE={{aws_profile}} AWS_REGION={{aws_region}} AWS_CMD={{aws}} scripts/sync_secrets_to_ssm.sh {{env_file}}
