"""An in-process Slurm stand-in, persisted to a JSON file.

Jobs move PENDING -> RUNNING -> (outcome) one step per ``tick()``, respecting
``afterany``/``afterok`` dependencies. Outcomes can be scripted per job name to
test failure handling. Because state lives in a file, a "crashed" engine and a
fresh one see the same jobs.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..core.fs import atomic_write_json, read_json
from .jobspec import JobSpec
from .sbatch import array_size
from .slurm import JobState, JobStatus


class FakeSlurm:
    def __init__(self, state_file: Path | None = None, *, run_commands: bool = False) -> None:
        self.state_file = Path(state_file) if state_file else None
        self.run_commands = run_commands
        self._data: dict[str, Any] = {"next_id": 1000, "jobs": {}, "outcomes": []}
        if self.state_file and self.state_file.exists():
            self._data = read_json(self.state_file)

    # -- persistence -------------------------------------------------------
    def _save(self) -> None:
        if self.state_file:
            atomic_write_json(self.state_file, self._data)

    def _reload(self) -> None:
        if self.state_file and self.state_file.exists():
            self._data = read_json(self.state_file)

    # -- scripting failures -------------------------------------------------
    def set_outcome(self, name_pattern: str, state: JobState, exit_code: int = 1) -> None:
        """Jobs whose name matches ``name_pattern`` (glob) will finish in ``state``."""
        self._reload()
        self._data["outcomes"].append({"pattern": name_pattern, "state": state.value, "exit_code": exit_code})
        self._save()

    def _outcome_for(self, name: str) -> tuple[JobState, int]:
        for rule in reversed(self._data["outcomes"]):
            if fnmatch.fnmatch(name, rule["pattern"]):
                return JobState(rule["state"]), int(rule["exit_code"])
        return JobState.COMPLETED, 0

    # -- SlurmClient protocol ------------------------------------------------
    def submit(self, spec: JobSpec) -> str:
        self._reload()
        job_id = str(self._data["next_id"])
        self._data["next_id"] += 1
        self._data["jobs"][job_id] = {
            "spec": spec.model_dump(mode="json"),
            "state": JobState.PENDING.value,
            "exit_code": None,
            "reason": "",
        }
        self._save()
        return job_id

    def status(self, job_id: str) -> JobStatus:
        self._reload()
        job = self._data["jobs"].get(str(job_id))
        if job is None:
            return JobStatus(job_id=str(job_id), state=JobState.UNKNOWN, reason="no such job")
        return JobStatus(
            job_id=str(job_id),
            state=JobState(job["state"]),
            exit_code=job["exit_code"],
            reason=job["reason"],
            name=job["spec"]["name"],
        )

    def status_many(self, job_ids: Iterable[str]) -> dict[str, JobStatus]:
        return {str(j): self.status(str(j)) for j in job_ids}

    def cancel(self, job_id: str) -> None:
        self._reload()
        job = self._data["jobs"].get(str(job_id))
        if job and not JobState(job["state"]).is_terminal:
            job["state"] = JobState.CANCELLED.value
            job["reason"] = "cancelled"
            self._save()

    def estimate_start(self, spec: JobSpec) -> float | None:
        gpu_type = spec.gpus.type if spec.gpus else None
        return self._data.get("estimates", {}).get(f"{spec.partition}:{gpu_type}")

    def set_estimate(self, partition: str, gpu_type: str | None, start: float | None) -> None:
        """Script what ``estimate_start`` answers for jobs on this partition and GPU type."""
        self._reload()
        self._data.setdefault("estimates", {})[f"{partition}:{gpu_type}"] = start
        self._save()

    # -- simulation ------------------------------------------------------------
    @property
    def jobs(self) -> dict[str, dict[str, Any]]:
        self._reload()
        return self._data["jobs"]

    def _deps_state(self, job: dict[str, Any]) -> str:
        spec = job["spec"]
        deps = [self._data["jobs"].get(d) for d in spec.get("dependency", [])]
        if any(d is None for d in deps):
            return "broken"
        states = [JobState(d["state"]) for d in deps]  # type: ignore[index]
        if not all(s.is_terminal for s in states):
            return "waiting"
        if spec.get("dependency_type") == "afterok" and any(s is not JobState.COMPLETED for s in states):
            return "broken"
        return "ok"

    def tick(self, steps: int = 1) -> None:
        for _ in range(steps):
            self._reload()
            for job_id, job in self._data["jobs"].items():
                state = JobState(job["state"])
                if state is JobState.PENDING:
                    deps = self._deps_state(job)
                    if deps == "ok":
                        job["state"] = JobState.RUNNING.value
                    elif deps == "broken":
                        job["state"] = JobState.CANCELLED.value
                        job["reason"] = "DependencyNeverSatisfied"
                elif state is JobState.RUNNING:
                    outcome, exit_code = self._outcome_for(job["spec"]["name"])
                    if outcome is JobState.COMPLETED and self.run_commands:
                        outcome, exit_code = self._execute(job_id, job["spec"])
                    job["state"] = outcome.value
                    job["exit_code"] = exit_code
            self._save()

    def finish_all(self, max_steps: int = 50) -> None:
        """Tick until no job is pending or running."""
        for _ in range(max_steps):
            if all(JobState(j["state"]).is_terminal for j in self.jobs.values()):
                return
            self.tick()

    @staticmethod
    def _execute(job_id: str, spec: dict[str, Any]) -> tuple[JobState, int]:
        """Run the job's command here, once per array task, with the job's environment.

        Output goes where the real client's scripts put it (``<workdir>/.rtool/logs/<name>-<job>[_<task>].out``).
        """
        command = spec["command"]
        env = os.environ | dict(spec.get("env") or {}) | {"SLURM_JOB_ID": job_id}
        tasks = range(array_size(spec["array"])) if spec.get("array") else [None]
        logs = Path(spec["workdir"]) / ".rtool" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        worst = 0
        for task in tasks:
            suffix = ""
            if task is not None:
                env |= {"SLURM_ARRAY_JOB_ID": job_id, "SLURM_ARRAY_TASK_ID": str(task)}
                suffix = f"_{task}"
            result = subprocess.run(
                command,
                shell=isinstance(command, str),
                cwd=spec["workdir"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=600,
                env=env,
            )
            (logs / f"{spec['name']}-{job_id}{suffix}.out").write_text(result.stdout or "", encoding="utf-8")
            worst = worst or result.returncode
        return (JobState.COMPLETED if worst == 0 else JobState.FAILED), worst
