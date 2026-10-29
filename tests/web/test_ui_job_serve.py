"""`rtool ui --mode job --serve` (what the UI job runs): a private Unix socket, the token in a file, clean exit."""

from __future__ import annotations

import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from research_tool.core.fs import read_json  # noqa: E402
from research_tool.service.ui_job import ENDPOINT_FILE, SOCKET_FILE, TOKEN_FILE  # noqa: E402

RTOOL = Path(sys.executable).parent / "rtool"


@pytest.fixture
def short_dir() -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix="rt-", dir="/tmp"))  # Unix socket paths are short
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.skipif(not RTOOL.exists(), reason="rtool is not installed next to this Python")
def test_job_serves_on_a_private_socket_and_cleans_up(short_dir: Path) -> None:
    runs = short_dir / "projects"
    runs.mkdir()
    config = short_dir / "cluster.yaml"
    config.write_text(f"paths: {{runs_root: {runs}}}\nslurm: {{backend: real}}\n", encoding="utf-8")
    state = short_dir / "ui"
    env = {**os.environ, "SLURM_JOB_ID": "4242", "RTOOL_CONFIG": str(config)}
    server = subprocess.Popen(
        [str(RTOOL), "ui", "--mode", "job", "--serve", "--state-dir", str(state)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not (state / ENDPOINT_FILE).exists():
            assert server.poll() is None, server.stdout.read() if server.stdout else ""
            assert time.monotonic() < deadline, "the server did not announce itself"
            time.sleep(0.1)
        endpoint = read_json(state / ENDPOINT_FILE)
        assert endpoint["job_id"] == "4242" and endpoint["bind"] == "socket" and endpoint["port"] is None
        sock = state / SOCKET_FILE
        assert endpoint["socket"] == str(sock)
        assert stat.S_IMODE(state.stat().st_mode) == 0o700
        for name in (SOCKET_FILE, TOKEN_FILE, ENDPOINT_FILE):
            assert stat.S_IMODE((state / name).stat().st_mode) == 0o600, name
        token = (state / TOKEN_FILE).read_text().strip()

        client = httpx.Client(transport=httpx.HTTPTransport(uds=str(sock)), base_url="http://127.0.0.1:8780")
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/runs").status_code == 401
        headers = {"X-Requested-With": "rtool", "Origin": "http://127.0.0.1:8780"}
        assert client.post("/api/session", json={"token": token}, headers=headers).status_code == 204
        settings = client.get("/api/settings").json()
        assert settings["mode"] == "job" and not settings["runs_locally"]
        assert client.get("/api/runs").json() == []
        client.close()

        server.send_signal(signal.SIGTERM)  # what Slurm sends at the time limit or on scancel
        out, _ = server.communicate(timeout=20)
        assert token not in out  # the job log never holds the token
        assert not any((state / n).exists() for n in (ENDPOINT_FILE, SOCKET_FILE, TOKEN_FILE))
    finally:
        if server.poll() is None:
            server.kill()
            server.wait()


def test_serve_is_only_for_the_job(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from research_tool.cli.main import app

    config = tmp_path / "cluster.yaml"
    config.write_text(f"paths: {{runs_root: {tmp_path / 'runs'}}}\nslurm: {{backend: real}}\n", encoding="utf-8")
    env = {"RTOOL_CONFIG": str(config), "SLURM_JOB_ID": ""}
    result = CliRunner().invoke(app, ["ui", "--mode", "job", "--serve", "--state-dir", str(tmp_path / "ui")], env=env)
    assert result.exit_code != 0 and "for the UI job itself" in result.output
    result = CliRunner().invoke(app, ["ui", "--mode", "job", "--status"])  # the tests' empty config: no cluster
    assert result.exit_code != 0 and "needs a cluster" in result.output
