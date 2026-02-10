from typer.testing import CliRunner

from data_bolt.botctl.main import app

runner = CliRunner()


def test_botctl_help_works() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "simulate" in result.output
    assert "chat" in result.output


def test_botctl_version_works() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip()
