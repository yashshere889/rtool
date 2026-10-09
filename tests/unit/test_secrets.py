"""API keys: loaded from a private file, never handed to experiment code, vLLM or Slurm jobs."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from research_tool.core import secrets
from research_tool.core.config import ExperimentsConfig, LLMConfig
from research_tool.core.secrets import SecretsError, load_secrets, secrets_status, without_secrets


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(secrets, "_loaded", set())


def write_secrets(tmp_path: Path, text: str, mode: int = 0o600) -> Path:
    path = tmp_path / "secrets.env"
    path.write_text(text)
    path.chmod(mode)
    return path


def test_load_secrets_fills_the_environment_without_overriding(tmp_path: Path) -> None:
    path = write_secrets(tmp_path, "# keys\nexport S2_API_KEY='s2-abc'\nOPENALEX_API_KEY=oa-1\nMY_LAB_TOKEN=x\n")
    environ = {"RTOOL_SECRETS": str(path), "OPENALEX_API_KEY": "from-shell"}
    assert load_secrets(environ) == ["MY_LAB_TOKEN", "OPENALEX_API_KEY", "S2_API_KEY"]
    assert environ["S2_API_KEY"] == "s2-abc" and environ["OPENALEX_API_KEY"] == "from-shell"


def test_a_readable_secrets_file_is_refused(tmp_path: Path) -> None:
    path = write_secrets(tmp_path, "S2_API_KEY=x\n", mode=0o644)
    with pytest.raises(SecretsError, match="chmod 600"):
        load_secrets({"RTOOL_SECRETS": str(path)})


def test_without_secrets_drops_known_and_loaded_names(tmp_path: Path) -> None:
    path = write_secrets(tmp_path, "MY_LAB_TOKEN=x\n")
    environ = {"RTOOL_SECRETS": str(path), "HF_TOKEN": "hf", "PATH": "/bin", "S2_API_KEY": "s2"}
    load_secrets(environ)
    assert without_secrets(environ) == {"PATH": "/bin"}  # MY_LAB_TOKEN came from the file, so it is hidden too
    assert secrets_status(environ) == {
        "HF_TOKEN": True,
        "OPENALEX_API_KEY": False,
        "S2_API_KEY": True,
        "UNPAYWALL_EMAIL": False,
    }


def test_experiment_code_and_vllm_never_see_the_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from research_tool.agents.coder.env import ExperimentEnv
    from research_tool.llm.server import VLLMServer

    monkeypatch.setenv("S2_API_KEY", "s2-secret")
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    experiment = ExperimentEnv(ExperimentsConfig(), tmp_path).environment(gpu=False)
    server = VLLMServer(LLMConfig(), tmp_path).environment("key")
    for env in (experiment, server):
        assert "S2_API_KEY" not in env and "HF_TOKEN" not in env
        assert "s2-secret" not in env.values()
        assert env.get("PATH")  # the rest of the environment is still there


def test_slurm_commands_run_without_the_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    from research_tool.hpc import slurm_client

    seen: dict[str, Any] = {}

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setenv("OPENALEX_API_KEY", "oa-secret")
    monkeypatch.setattr(slurm_client.subprocess, "run", fake_run)
    slurm_client.default_runner(["sbatch", "--parsable", "job.sh"], None, 5)
    assert "OPENALEX_API_KEY" not in seen["env"] and seen["env"].get("PATH")  # sbatch exports what it is given
