"""Environment-related commands for botctl."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import boto3
import typer
from dotenv import dotenv_values, set_key

DEFAULT_LAAS_SSM_PARAM = "/DATA/PIPELINE/API_KEY/OPENAI"


def sync_laas_key_command(
    env_file: Annotated[Path, typer.Option("--env-file", help="Target .env file path.")] = Path(
        ".env"
    ),
    ssm_param: Annotated[
        str, typer.Option("--ssm-param", help="SSM parameter name for LAAS API key.")
    ] = DEFAULT_LAAS_SSM_PARAM,
    region: Annotated[
        str | None, typer.Option("--region", help="AWS region for SSM lookup.")
    ] = None,
    profile: Annotated[
        str | None, typer.Option("--profile", help="AWS profile to use for SSM lookup.")
    ] = None,
    override_existing: Annotated[
        bool,
        typer.Option(
            "--override-existing/--keep-existing", help="Overwrite existing LAAS_API_KEY."
        ),
    ] = False,
) -> None:
    """Fetch LAAS API key from SSM and write it into .env as LAAS_API_KEY."""
    current = dotenv_values(env_file)
    if current.get("LAAS_API_KEY") and not override_existing:
        typer.echo(f"Skipped: LAAS_API_KEY already exists in {env_file}.")
        return

    if profile:
        session = boto3.Session(profile_name=profile)
    else:
        session = boto3.Session()
    client = session.client("ssm", region_name=region)
    response = client.get_parameter(Name=ssm_param, WithDecryption=True)
    parameter = response.get("Parameter", {})
    value = parameter.get("Value")
    if not isinstance(value, str) or not value:
        raise typer.Exit(code=1)

    env_file.touch(exist_ok=True)
    set_key(str(env_file), "LAAS_API_KEY", value, quote_mode="never")
    typer.echo(f"Wrote LAAS_API_KEY to {env_file} from {ssm_param}.")
