"""The UI as a Slurm job: submit or reuse, find where it serves, stop; and the private socket."""

from __future__ import annotations

import os
import shutil
import socket
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from research_tool.core.config import AppConfig, load_config
from research_tool.core.errors import RToolError
from research_tool.core.fs import atomic_write_json, read_json
from research_tool.hpc.fake_slurm import FakeSlurm
from research_tool.hpc.sbatch import render_script, validate_job
from research_tool.hpc.slurm import JobState
from research_tool.service.ui_job import (
    ENDPOINT_FILE,
    JOB_FILE,
    SOCKET_FILE,
    TOKEN_FILE,
    announce,
    find_ui_job,
    listen,
    private_dir,
    start_ui_job,
    stop_ui_job,
    ui_dir,
    ui_job_spec,
    wait_until_serving,
    withdraw,
)


@pytest.fixture
def short_dir() -> Iterator[Path]:
    """Unix socket paths are limited to ~104 bytes; pytest's tmp_path on macOS is longer."""
    path = Path(tempfile.mkdtemp(prefix="rt-", dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def cluster_config(runs_root: Path) -> AppConfig:
    return load_config(
        overrides={
            "paths": {"runs_root": str(runs_root)},
            "slurm": {
                "backend": "real",
                "submit_roots": [str(runs_root.parent)],
                "partitions": {
                    "nodes": {"max_time": "3-00:00:00"},
                    "gpu-l40s": {"gpu_type": "l40s", "gpus_per_node": 2},
                },
                "episode": {
                    "cpu_partition": "nodes",
                    "activate": "/venv/bin/activate",
                    "modules": ["miniforge3/25.3.0-python3.12.10"],
                    "config_files": ["~/Research tool/configs/barkla.yaml"],
                },
            },
        }
    )


def test_ui_dir_sits_beside_the_runs(tmp_path: Path) -> None:
    cfg = cluster_config(tmp_path / "rtool" / "projects")
    assert ui_dir(cfg) == tmp_path / "rtool" / "ui"
    assert ui_dir(cfg, tmp_path / "elsewhere") == tmp_path / "elsewhere"


def test_job_spec_is_a_small_isolated_cpu_job_without_secrets(tmp_path: Path) -> None:
    cfg = cluster_config(tmp_path / "rtool" / "projects")
    directory = tmp_path / "rtool" / "ui"
    spec = ui_job_spec(
        cfg, directory, hours=8, bind="socket", runs_root=tmp_path / "rtool" / "projects", config_files=[Path("x.yaml")]
    )
    assert spec.partition == "nodes" and spec.gpus is None and spec.ntasks == 1
    assert spec.time_limit == "08:00:00" and not spec.requeue
    assert validate_job(spec, cfg.slurm) == []
    command = str(spec.command)
    assert "unset PYTHONPATH" in command and "PYTHONNOUSERSITE=1" in command
    assert "source /venv/bin/activate" in command
    assert "rtool ui --mode job --serve --bind socket" in command
    assert f"--state-dir {directory}" in command
    assert "--config x.yaml" in command and "configs/barkla.yaml" in command
    script = render_script(spec, sentinel_dir=directory / "s", log_dir=directory / "l")
    assert "--no-requeue" in script and "token" not in script.lower()


def test_start_submits_once_then_reuses_and_stop_cancels(tmp_path: Path) -> None:
    cfg = cluster_config(tmp_path / "rtool" / "projects")
    directory = ui_dir(cfg)
    slurm = FakeSlurm()
    kwargs = {"hours": 2.0, "bind": "socket", "runs_root": tmp_path, "config_files": [], "user": "me"}

    job, new = start_ui_job(cfg, slurm, directory, **kwargs)  # type: ignore[arg-type]
    assert new and job.state == "PENDING" and not job.serving
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert read_json(directory / JOB_FILE)["job_id"] == job.job_id
    again, new = start_ui_job(cfg, slurm, directory, **kwargs)  # type: ignore[arg-type]
    assert not new and again.job_id == job.job_id
    assert len(slurm.jobs) == 1

    assert stop_ui_job(directory, slurm) == job.job_id
    assert find_ui_job(directory, slurm) is None
    assert stop_ui_job(directory, slurm) is None

    # a job that is gone leaves nothing behind for the next one to trip over
    (directory / TOKEN_FILE).write_text("old\n")
    (directory / ENDPOINT_FILE).write_text("{}")
    fresh, new = start_ui_job(cfg, slurm, directory, **kwargs)  # type: ignore[arg-type]
    assert new and fresh.job_id != job.job_id
    assert not (directory / TOKEN_FILE).exists() and not (directory / ENDPOINT_FILE).exists()


def test_serving_only_once_the_running_job_announced_itself(tmp_path: Path) -> None:
    directory = private_dir(tmp_path / "ui")
    slurm = FakeSlurm()
    cfg = cluster_config(tmp_path / "projects")
    job, _ = start_ui_job(cfg, slurm, directory, hours=1, bind="socket", runs_root=tmp_path, config_files=[], user="me")
    endpoint = {"job_id": job.job_id, "node": "node041", "bind": "socket", "socket": "/x/ui.sock", "port": None}
    atomic_write_json(directory / ENDPOINT_FILE, endpoint)
    assert not find_ui_job(directory, slurm).serving  # type: ignore[union-attr]  # still PENDING
    slurm.tick()
    found = find_ui_job(directory, slurm)
    assert found is not None and found.state == JobState.RUNNING.value
    assert found.serving and found.node == "node041" and found.socket == "/x/ui.sock"
    atomic_write_json(directory / ENDPOINT_FILE, {**endpoint, "job_id": "999"})  # an earlier job's
    assert not find_ui_job(directory, slurm).serving  # type: ignore[union-attr]


def test_wait_until_serving(tmp_path: Path) -> None:
    directory = private_dir(tmp_path / "ui")
    slurm = FakeSlurm()
    cfg = cluster_config(tmp_path / "projects")
    job, _ = start_ui_job(
        cfg, slurm, directory, hours=1, bind="cluster", runs_root=tmp_path, config_files=[], user="me"
    )
    seen: list[str] = []

    def step(_: float) -> None:  # the job starts and announces itself while we wait
        slurm.tick()
        atomic_write_json(directory / ENDPOINT_FILE, {"job_id": job.job_id, "node": "node1", "port": 4242})

    served = wait_until_serving(directory, slurm, timeout=60, on_wait=lambda j: seen.append(j.state), sleep=step)
    assert served is not None and served.port == 4242 and seen == ["PENDING"]
    assert wait_until_serving(directory, slurm, timeout=0, sleep=step) is not None

    slurm.cancel(job.job_id)
    with pytest.raises(RToolError, match="ended before it served"):
        wait_until_serving(directory, slurm, timeout=1, sleep=step)


def test_still_queued_after_the_timeout(tmp_path: Path) -> None:
    directory = private_dir(tmp_path / "ui")
    slurm = FakeSlurm()
    cfg = cluster_config(tmp_path / "projects")
    start_ui_job(cfg, slurm, directory, hours=1, bind="socket", runs_root=tmp_path, config_files=[], user="me")
    assert wait_until_serving(directory, slurm, timeout=0, sleep=lambda _: None) is None


def test_private_dir_tightens_permissions(tmp_path: Path) -> None:
    loose = tmp_path / "ui"
    loose.mkdir(mode=0o755)
    os.chmod(loose, 0o755)
    private_dir(loose)
    assert stat.S_IMODE(loose.stat().st_mode) == 0o700


def test_socket_only_the_owner_can_open_and_the_endpoint(short_dir: Path) -> None:
    directory = private_dir(short_dir / "ui")
    (directory / SOCKET_FILE).write_text("stale")  # left by a job that died
    sock, where = listen(directory, "socket")
    try:
        path = directory / SOCKET_FILE
        assert where == {"socket": str(path), "port": None}
        assert stat.S_ISSOCK(path.stat().st_mode) and stat.S_IMODE(path.stat().st_mode) == 0o600
        sock.listen(1)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(path))
        client.close()

        announce(directory, "123", "socket", where)
        endpoint = read_json(directory / ENDPOINT_FILE)
        assert endpoint["job_id"] == "123" and endpoint["socket"] == str(path) and endpoint["node"]
        assert stat.S_IMODE((directory / ENDPOINT_FILE).stat().st_mode) == 0o600
        withdraw(directory, "999")  # not ours: kept
        assert (directory / ENDPOINT_FILE).exists()
        (directory / TOKEN_FILE).write_text("t\n")
        withdraw(directory, "123")
        assert not any((directory / n).exists() for n in (ENDPOINT_FILE, SOCKET_FILE, TOKEN_FILE))
    finally:
        sock.close()


def test_cluster_bind_listens_on_a_free_port(tmp_path: Path) -> None:
    sock, where = listen(private_dir(tmp_path / "ui"), "cluster")
    try:
        assert where["socket"] is None and isinstance(where["port"], int) and where["port"] > 0
    finally:
        sock.close()
