"""Slurm client that shells out to sbatch / squeue / sacct / scancel.

Job status is resolved in three layers, so it works even where accounting is
restricted:

1. ``squeue -u $USER -r`` for jobs still in the queue (never errors on finished ids);
2. ``sacct`` for finished jobs, if accounting is readable;
3. sentinel files the job script writes (``.started`` / ``.exit``).

Every submission is recorded under ``<state_dir>/registry/<job_id>.json`` (one
file per job, so concurrent submitters never clobber each other).
"""

from __future__ import annotations

import getpass
import re
import secrets
import subprocess
import time
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.config import SlurmConfig
from ..core.errors import RToolError
from ..core.fs import atomic_write_json, atomic_write_text, read_json
from ..core.secrets import without_secrets
from .jobspec import JobSpec
from .sbatch import JobSpecError, array_size, render_script, validate_job
from .slurm import JobState, JobStatus

Runner = Callable[[list[str], Path | None, float], "subprocess.CompletedProcess[str]"]

_SQUEUE_STATES = {
    "PENDING": JobState.PENDING,
    "CONFIGURING": JobState.RUNNING,
    "RUNNING": JobState.RUNNING,
    "COMPLETING": JobState.COMPLETING,
    "SUSPENDED": JobState.PENDING,
    "REQUEUED": JobState.PENDING,
    "REQUEUE_HOLD": JobState.PENDING,
    "RESIZING": JobState.RUNNING,
    "STAGE_OUT": JobState.COMPLETING,
    "PREEMPTED": JobState.PENDING,  # still in the queue: about to be requeued
}
# Worst outcome wins when aggregating array tasks.
_SEVERITY = [
    JobState.COMPLETED,
    JobState.CANCELLED,
    JobState.PREEMPTED,
    JobState.NODE_FAIL,
    JobState.FAILED,
    JobState.TIMEOUT,
    JobState.OUT_OF_MEMORY,
]
SUBMIT_GRACE_S = 120  # a just-submitted job may not be visible to squeue yet
_TEST_ONLY = re.compile(r"to start at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")


class SlurmCommandError(RToolError):
    """A Slurm command was missing, timed out or failed."""


def default_runner(args: list[str], cwd: Path | None, timeout: float) -> subprocess.CompletedProcess[str]:
    # sbatch exports this environment into the job; jobs that need API keys (episodes) load them themselves
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False, env=without_secrets()
    )


def parse_sacct_state(text: str) -> JobState:
    word = text.strip().split()[0] if text.strip() else "UNKNOWN"
    word = word.rstrip("+")
    try:
        return JobState(word)
    except ValueError:
        return _SQUEUE_STATES.get(word, JobState.UNKNOWN)


def _base_id(job_id: str) -> str:
    return job_id.split("_", 1)[0].split(".", 1)[0]


class SubprocessSlurm:
    def __init__(
        self,
        config: SlurmConfig,
        *,
        state_dir: Path,
        runner: Runner = default_runner,
        user: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.state_dir = Path(state_dir)
        self.runner = runner
        self.user = user or getpass.getuser()
        self.clock = clock
        self.sacct_available: bool | None = None

    # ------------------------------------------------------------- plumbing
    def _run(self, args: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
        try:
            result = self.runner(args, cwd, self.config.command_timeout_s)
        except FileNotFoundError:
            raise SlurmCommandError(f"{args[0]} not found; is this a Slurm login or compute node?") from None
        except subprocess.TimeoutExpired:
            raise SlurmCommandError(f"{args[0]} did not answer within {self.config.command_timeout_s:.0f}s") from None
        if check and result.returncode != 0:
            raise SlurmCommandError(f"{' '.join(args[:2])} failed ({result.returncode}): {result.stderr.strip()[:500]}")
        return result

    def _registry(self, job_id: str) -> Path:
        return self.state_dir / "registry" / f"{_base_id(job_id)}.json"

    def _record(self, job_id: str) -> dict[str, Any] | None:
        path = self._registry(job_id)
        return read_json(path) if path.exists() else None

    # --------------------------------------------------------------- submit
    def queue_size(self) -> int:
        out = self._run(["squeue", "-h", "-u", self.user, "-r", "-o", "%i"]).stdout
        return sum(1 for line in out.splitlines() if line.strip())

    def _write_script(self, spec: JobSpec) -> tuple[Path, Path]:
        base = Path(spec.workdir) / ".rtool"
        sentinel_dir, log_dir = base / "sentinels", base / "logs"
        sentinel_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)  # Slurm does not create the -o directory
        script = base / "scripts" / f"{spec.name}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}.sh"
        atomic_write_text(script, render_script(spec, sentinel_dir=sentinel_dir, log_dir=log_dir))
        return script, sentinel_dir

    def estimate_start(self, spec: JobSpec) -> float | None:
        problems = validate_job(spec, self.config)
        if problems:
            raise JobSpecError("; ".join(problems))
        script, _ = self._write_script(spec)
        try:
            result = self._run(["sbatch", "--test-only", str(script)], cwd=Path(spec.workdir), check=False)
        finally:
            script.unlink(missing_ok=True)
        # "sbatch: Job 123 to start at 2026-10-08T15:30:02 using 24 processors on nodes gpu47 in partition ..."
        match = _TEST_ONLY.search(result.stderr + result.stdout)
        if match is None:
            return None
        try:
            return datetime.fromisoformat(match.group(1)).timestamp()  # Slurm prints local time
        except ValueError:
            return None

    def submit(self, spec: JobSpec, *, allowed_partitions: Iterable[str] | None = None) -> str:
        problems = validate_job(spec, self.config, allowed_partitions=allowed_partitions)
        if problems:
            raise JobSpecError("; ".join(problems))
        queued = self.queue_size()
        if queued + array_size(spec.array) > self.config.max_jobs_in_queue:
            raise JobSpecError(
                f"{queued} jobs already queued; adding {array_size(spec.array)} would pass the limit "
                f"of {self.config.max_jobs_in_queue}"
            )
        script, sentinel_dir = self._write_script(spec)
        out = self._run(["sbatch", "--parsable", str(script)], cwd=Path(spec.workdir)).stdout.strip()
        job_id = out.split(";", 1)[0]
        if not job_id.isdigit():
            raise SlurmCommandError(f"unexpected sbatch output: {out!r}")
        atomic_write_json(
            self._registry(job_id),
            {
                "job_id": job_id,
                "name": spec.name,
                "array": bool(spec.array),
                "sentinel_dir": str(sentinel_dir),
                "script": str(script),
                "submitted_at": self.clock(),
                "gpu_hours_requested": spec.gpu_hours_requested * array_size(spec.array),
                "metadata": spec.metadata,
            },
        )
        return job_id

    # --------------------------------------------------------------- status
    def _active(self) -> dict[str, tuple[JobState, str]]:
        out = self._run(["squeue", "-h", "-u", self.user, "-r", "-o", "%i|%T|%r"]).stdout
        active: dict[str, tuple[JobState, str]] = {}
        for line in out.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            base = _base_id(parts[0])
            state = _SQUEUE_STATES.get(parts[1], JobState.PENDING)
            reason = parts[2] if len(parts) > 2 else ""
            previous = active.get(base)
            if previous is None or (previous[0] is JobState.PENDING and state is not JobState.PENDING):
                active[base] = (state, reason)
        return active

    def _sacct(self, job_ids: list[str]) -> dict[str, JobStatus]:
        if self.sacct_available is False or not job_ids:
            return {}
        result = self._run(
            ["sacct", "-X", "-n", "-P", "-j", ",".join(job_ids), "-o", "JobID,State,ExitCode"], check=False
        )
        if result.returncode != 0:
            self.sacct_available = False
            return {}
        self.sacct_available = True
        tasks: dict[str, list[tuple[JobState, int]]] = {}
        for line in result.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 3:
                continue
            code_text = parts[2].split(":", 1)[0]
            code = int(code_text) if code_text.isdigit() else 0
            tasks.setdefault(_base_id(parts[0]), []).append((parse_sacct_state(parts[1]), code))
        statuses: dict[str, JobStatus] = {}
        for job_id, entries in tasks.items():
            states = [s for s, _ in entries]
            unfinished = [s for s in states if not s.is_terminal]
            if unfinished:
                state = JobState.RUNNING if JobState.RUNNING in unfinished else unfinished[0]
            else:
                state = max(states, key=lambda s: _SEVERITY.index(s) if s in _SEVERITY else 0)
            statuses[job_id] = JobStatus(
                job_id=job_id, state=state, exit_code=max(c for _, c in entries), reason="sacct"
            )
        return statuses

    def _from_sentinels(self, job_id: str) -> JobStatus:
        record = self._record(job_id)
        if record is None:
            return JobStatus(job_id=job_id, state=JobState.FAILED, reason="unknown job (not submitted by rtool here)")
        directory = Path(record["sentinel_dir"])
        pattern = f"{job_id}_*" if record.get("array") else job_id
        exits = sorted(directory.glob(f"{pattern}.exit"))
        started = sorted(directory.glob(f"{pattern}.started"))
        name = record.get("name", "")
        if exits:
            codes = []
            for path in exits:
                first = path.read_text(encoding="utf-8").split()
                codes.append(int(first[0]) if first and first[0].lstrip("-").isdigit() else 1)
            if record.get("array") and len(exits) < len(started):
                return JobStatus(
                    job_id=job_id,
                    state=JobState.FAILED,
                    exit_code=max(codes),
                    name=name,
                    reason="some array tasks ended without an exit record",
                )
            worst = max(codes)
            state = JobState.COMPLETED if worst == 0 else JobState.FAILED
            return JobStatus(job_id=job_id, state=state, exit_code=worst, name=name, reason="sentinel")
        if started:
            return JobStatus(
                job_id=job_id,
                state=JobState.FAILED,
                name=name,
                reason="ended without an exit record (time limit, cancellation or node failure)",
            )
        if self.clock() - float(record.get("submitted_at", 0)) < SUBMIT_GRACE_S:
            return JobStatus(job_id=job_id, state=JobState.PENDING, name=name, reason="just submitted")
        return JobStatus(job_id=job_id, state=JobState.CANCELLED, name=name, reason="left the queue without running")

    def status_many(self, job_ids: Iterable[str]) -> dict[str, JobStatus]:
        ids = [str(j) for j in job_ids]
        active = self._active()
        result: dict[str, JobStatus] = {}
        for job_id in ids:
            if job_id in active:
                state, reason = active[job_id]
                result[job_id] = JobStatus(job_id=job_id, state=state, reason=reason)
        finished = self._sacct([j for j in ids if j not in result])
        for job_id in ids:
            if job_id not in result:
                result[job_id] = finished.get(job_id) or self._from_sentinels(job_id)
        return result

    def status(self, job_id: str) -> JobStatus:
        return self.status_many([job_id])[str(job_id)]

    def cancel(self, job_id: str) -> None:
        self._run(["scancel", str(job_id)], check=False)
