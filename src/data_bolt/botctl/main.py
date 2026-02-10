"""botctl CLI entrypoint."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import typer

from data_bolt.botctl.chat import chat_command
from data_bolt.botctl.env import sync_laas_key_command
from data_bolt.botctl.simulate import simulate_command

app = typer.Typer(help="Data Bolt local tooling CLI.")

app.command("simulate")(simulate_command)
app.command("chat")(chat_command)
app.command("sync-laas-key")(sync_laas_key_command)


@app.command("version")
def version_command() -> None:
    """Print installed package version."""
    try:
        ver = version("data_bolt")
    except PackageNotFoundError:
        ver = "0.0.0+local"
    typer.echo(ver)


if __name__ == "__main__":
    app()
