from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from typer.testing import CliRunner

from data_bolt.botctl.env import sync_laas_key_command
from data_bolt.botctl.main import app

runner = CliRunner()


def test_sync_laas_key_command_writes_env(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"

    class FakeClient:
        def get_parameter(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["Name"] == "/p/key"
            return {"Parameter": {"Value": "from-ssm"}}

    class FakeSession:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs.get("profile_name") == "sandbox"

        def client(self, service_name: str, region_name: str | None = None) -> Any:
            assert service_name == "ssm"
            assert region_name == "ap-northeast-2"
            return FakeClient()

    monkeypatch.setattr("data_bolt.botctl.env.boto3.Session", FakeSession)

    sync_laas_key_command(
        env_file=env_path,
        ssm_param="/p/key",
        region="ap-northeast-2",
        profile="sandbox",
        override_existing=False,
    )

    values = dotenv_values(env_path)
    assert values["LAAS_API_KEY"] == "from-ssm"


def test_sync_laas_key_keeps_existing_value_by_default(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("LAAS_API_KEY=existing\n", encoding="utf-8")

    class FakeClient:
        def get_parameter(self, **kwargs: Any) -> dict[str, Any]:
            return {"Parameter": {"Value": "from-ssm"}}

    class FakeSession:
        def client(self, service_name: str, region_name: str | None = None) -> Any:
            return FakeClient()

    monkeypatch.setattr("data_bolt.botctl.env.boto3.Session", FakeSession)

    sync_laas_key_command(env_file=env_path, override_existing=False)
    values = dotenv_values(env_path)
    assert values["LAAS_API_KEY"] == "existing"


def test_simulate_loads_env_file_before_run(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("LAAS_API_KEY=env-value\n", encoding="utf-8")

    def fake_direct(payload: dict[str, Any], _trace_enabled: bool) -> dict[str, Any]:
        import os

        assert os.environ.get("LAAS_API_KEY") == "env-value"
        return {
            "mode": "direct",
            "payload": payload,
            "trace": [],
            "result": {
                "action": "chat_reply",
                "should_respond": True,
                "candidate_sql": None,
                "response_text": "ok",
            },
        }

    monkeypatch.setattr("data_bolt.botctl.simulate._run_direct_with_trace", fake_direct)
    monkeypatch.delenv("LAAS_API_KEY", raising=False)

    result = runner.invoke(
        app,
        ["simulate", "--text", "안녕하세요", "--env-file", str(env_path)],
    )
    assert result.exit_code == 0
