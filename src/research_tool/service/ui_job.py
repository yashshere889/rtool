"""The web UI as a small Slurm job (``rtool ui --mode job``): start or reuse it, find it, stop it.

Barkla forbids resident processes on login nodes, so the UI server runs as a CPU job. By
default it listens on a Unix socket in a folder only you can open, so nobody else on the
cluster can reach it, token or not; ``scripts/rtool-ui`` on the Mac tunnels to it through
the login node. ``bind="cluster"`` listens on a TCP port on the cluster network instead
(token-protected), for sites where ssh to compute nodes is not allowed.

Files in the UI folder (scratch, mode 0700):

* ``job.json``      the job id, written when the job is submitted;
* ``endpoint.json`` where the server listens, written by the job once it serves;
* ``token``         the access token (0600), written by the job: never in argv, the
                    environment, the job script or a log.
"""

from __future__ import annotations

import os
import shlex
import socket
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..core.config import AppConfig
from ..core.errors import RToolError
from ..core.fs import atomic_write_json, now_iso, read_json
from ..hpc.episode import ISOLATE_PYTHON
from ..hpc.jobspec import JobSpec, format_time_limit
from ..hpc.slurm import JobState, SlurmClient

Bind = Literal["socket", "cluster"]

JOB_FILE = "job.json"
ENDPOINT_FILE = "endpoint.json"
TOKEN_FILE = "token"
SOCKET_FILE = "ui.sock"
JOB_NAME = "rtool-ui"


class UIJob(BaseModel):
    """The UI job as the login node sees it (no token)."""

    job_id: str
    state: str  # Slurm's: PENDING, RUNNING, ...
    reason: str = ""  # why it is pending
    bind: Bind = "socket"
    hours: float = 8
    submitted_at: str = ""
    directory: str
    user: str
    # once it serves
    serving: bool = False
    node: str | None = None
    socket: str | None = None
    port: int | None = None
    started_at: str | None = None


def ui_dir(config: AppConfig, override: Path | None = None) -> Path:
    """Next to the runs (scratch on Barkla: jobs must be submitted from there)."""
    return (override or config.paths.resolved_runs_root().parent / "ui").expanduser()


def private_dir(path: Path) -> Path:
    """Create ``path`` if needed and make sure only its owner can open it."""
    path.mkdir(parents=True, exist_ok=True)
    info = path.stat()
    if info.st_uid != os.getuid():
        raise RToolError(f"{path} belongs to someone else; choose another --state-dir")
    if stat.S_IMODE(info.st_mode) != 0o700:
        os.chmod(path, 0o700)
    return path


def ui_job_spec(
    config: AppConfig,
    directory: Path,
    *,
    hours: float,
    bind: Bind,
    runs_root: Path,
    config_files: list[Path],
) -> JobSpec:
    ep = config.slurm.episode
    activate = f"source {shlex.quote(os.path.expandvars(ep.activate))} && " if ep.activate else ""
    files = [os.path.expanduser(str(p)) for p in [*ep.config_files, *config_files]]
    config_args = "".join(f" --config {shlex.quote(p)}" for p in dict.fromkeys(files))
    command = (
        f"{ISOLATE_PYTHON}{activate}exec rtool ui --mode job --serve --bind {bind}"
        f" --state-dir {shlex.quote(str(directory))} --runs-root {shlex.quote(str(runs_root))}{config_args}"
    )
    return JobSpec(
        name=JOB_NAME,
        partition=ep.cpu_partition,
        command=command,
        workdir=directory,
        time_limit=format_time_limit(hours * 3600),
        ntasks=1,
        modules=list(ep.modules),
        requeue=False,  # a new start makes a new token; the launcher reconnects instead
        metadata={"kind": "ui"},
    )


def _alive(slurm: SlurmClient, job_id: str) -> tuple[JobState, str] | None:
    status = slurm.status(job_id)
    return None if status.state.is_terminal else (status.state, status.reason)


def find_ui_job(directory: Path, slurm: SlurmClient) -> UIJob | None:
    """The UI job that is queued or running, if any."""
    job_path = directory / JOB_FILE
    if not job_path.exists():
        return None
    job = read_json(job_path)
    alive = _alive(slurm, str(job["job_id"]))
    if alive is None:
        return None
    state, reason = alive
    found = UIJob(
        job_id=str(job["job_id"]),
        state=state.value,
        reason=reason,
        bind=job.get("bind", "socket"),
        hours=job.get("hours", 8),
        submitted_at=job.get("submitted_at", ""),
        directory=str(directory),
        user=job.get("user", ""),
    )
    endpoint_path = directory / ENDPOINT_FILE
    if state is JobState.RUNNING and endpoint_path.exists():
        endpoint = read_json(endpoint_path)
        if str(endpoint.get("job_id")) == found.job_id:
            found.serving = True
            found.node = endpoint.get("node")
            found.socket = endpoint.get("socket")
            found.port = endpoint.get("port")
            found.started_at = endpoint.get("started_at")
    return found


def start_ui_job(
    config: AppConfig,
    slurm: SlurmClient,
    directory: Path,
    *,
    hours: float,
    bind: Bind,
    runs_root: Path,
    config_files: list[Path],
    user: str,
) -> tuple[UIJob, bool]:
    """Reuse the queued or running UI job, or submit one. Returns (job, newly submitted)."""
    private_dir(directory)
    current = find_ui_job(directory, slurm)
    if current is not None:
        return current, False
    for name in (ENDPOINT_FILE, TOKEN_FILE, SOCKET_FILE):  # left by a job that is gone
        (directory / name).unlink(missing_ok=True)
    spec = ui_job_spec(config, directory, hours=hours, bind=bind, runs_root=runs_root, config_files=config_files)
    job_id = slurm.submit(spec)
    atomic_write_json(
        directory / JOB_FILE,
        {"job_id": job_id, "bind": bind, "hours": hours, "submitted_at": now_iso(), "user": user},
    )
    return (
        UIJob(job_id=job_id, state="PENDING", bind=bind, hours=hours, directory=str(directory), user=user),
        True,
    )


def wait_until_serving(
    directory: Path,
    slurm: SlurmClient,
    *,
    timeout: float,
    poll: float = 3.0,
    on_wait: Callable[[UIJob], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> UIJob | None:
    """Wait for the job to serve; None if it is still queued after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        job = find_ui_job(directory, slurm)
        if job is None:
            raise RToolError(f"the UI job ended before it served; see its log in {directory / '.rtool' / 'logs'}")
        if job.serving:
            return job
        if on_wait is not None:
            on_wait(job)
        if time.monotonic() >= deadline:
            return None
        sleep(poll)


def stop_ui_job(directory: Path, slurm: SlurmClient) -> str | None:
    """Cancel the UI job; returns its id, or None if none was queued or running."""
    job = find_ui_job(directory, slurm)
    if job is None:
        return None
    slurm.cancel(job.job_id)
    return job.job_id


# ----------------------------------------------------------------- inside the job


def listen(directory: Path, bind: Bind) -> tuple[socket.socket, dict[str, object]]:
    """The server's socket, bound before anything is announced, and what to announce."""
    if bind == "socket":
        path = directory / SOCKET_FILE
        path.unlink(missing_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(path))
        os.chmod(path, 0o600)
        return sock, {"socket": str(path), "port": None}
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 0))
    return sock, {"socket": None, "port": sock.getsockname()[1]}


def announce(directory: Path, job_id: str, bind: Bind, where: dict[str, object]) -> None:
    atomic_write_json(
        directory / ENDPOINT_FILE,
        {
            "job_id": job_id,
            "node": socket.gethostname().split(".", 1)[0],
            "bind": bind,
            **where,
            "started_at": now_iso(),
            "pid": os.getpid(),
        },
    )
    os.chmod(directory / ENDPOINT_FILE, 0o600)


def withdraw(directory: Path, job_id: str) -> None:
    """Remove the endpoint, socket and token when the server stops (if they are still this job's)."""
    endpoint = directory / ENDPOINT_FILE
    try:
        if endpoint.exists() and str(read_json(endpoint).get("job_id")) == job_id:
            for name in (ENDPOINT_FILE, SOCKET_FILE, TOKEN_FILE):
                (directory / name).unlink(missing_ok=True)
    except (OSError, ValueError):
        pass
