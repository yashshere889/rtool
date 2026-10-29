"""Who advances a run after a decision: Slurm episode jobs on the cluster, a worker thread locally."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from ..core.actions import ActionInbox
from ..core.run import RunHandle
from ..core.services import build_services, build_slurm
from ..hpc.episode import EpisodeTracker
from ..service import KickResult, kick

log = logging.getLogger(__name__)


class WorkerState(BaseModel):
    """Is something advancing this run right now?"""

    active: bool = False
    detail: str = ""  # e.g. "episode job 12345 (PENDING)", "working locally"
    job_id: str | None = None
    last_error: str | None = None


class Runner(Protocol):
    def wake(self, run: RunHandle, reason: str, *, needs_llm: bool = False) -> KickResult: ...

    def state(self, run: RunHandle) -> WorkerState: ...


class ClusterRunner:
    """Real Slurm: decisions submit (CPU) episode jobs, exactly like the CLI."""

    def __init__(self, cache_seconds: float = 15.0) -> None:
        self.cache_seconds = cache_seconds
        self._cache: dict[Path, tuple[float, WorkerState]] = {}
        self._lock = threading.Lock()

    def wake(self, run: RunHandle, reason: str, *, needs_llm: bool = False) -> KickResult:
        result = kick(run, reason, needs_llm=needs_llm)
        with self._lock:
            self._cache.pop(run.root, None)
        return result

    def state(self, run: RunHandle) -> WorkerState:
        """Asks Slurm at most every ``cache_seconds`` per run (squeue is not free on a busy cluster)."""
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(run.root)
            if cached is not None and now - cached[0] < self.cache_seconds:
                return cached[1]
        try:
            current = EpisodeTracker(run, build_slurm(run)).current()
        except Exception as exc:  # noqa: BLE001 - a Slurm hiccup must not break the page
            state = WorkerState(last_error=f"could not ask Slurm: {exc}")
        else:
            state = (
                WorkerState(
                    active=True,
                    job_id=str(current["job_id"]),
                    detail=f"episode job {current['job_id']} ({current.get('state', '?')})",
                )
                if current
                else WorkerState()
            )
        with self._lock:
            self._cache[run.root] = (now, state)
        return state


class LocalRunner:
    """No cluster: episodes run in a background thread of the server (one per run at a time).

    Runs created by the demo builder use its fakes; others use the run's own
    config (mock LLM, fake Slurm). Fake Slurm jobs are run and finished on the way.
    """

    def __init__(self) -> None:
        self._threads: dict[Path, threading.Thread] = {}
        self._errors: dict[Path, str] = {}
        self._lock = threading.Lock()

    def wake(self, run: RunHandle, reason: str, *, needs_llm: bool = False) -> KickResult:
        with self._lock:
            thread = self._threads.get(run.root)
            if thread is not None and thread.is_alive():
                return KickResult(
                    submitted=False, detail="the local worker is busy with this run; it will pick this up"
                )
            thread = threading.Thread(target=self._work, args=(run.root,), name=f"rtool-{run.run_id}", daemon=True)
            self._threads[run.root] = thread
            self._errors.pop(run.root, None)
            thread.start()
        return KickResult(submitted=True, detail="working on it locally")

    def state(self, run: RunHandle) -> WorkerState:
        with self._lock:
            thread = self._threads.get(run.root)
            active = thread is not None and thread.is_alive()
            error = self._errors.get(run.root)
        return WorkerState(active=active, detail="working locally" if active else "", last_error=error)

    def join(self, run: RunHandle, timeout: float | None = None) -> None:
        """Wait for the worker (tests, scripts)."""
        with self._lock:
            thread = self._threads.get(run.root)
        if thread is not None:
            thread.join(timeout)

    def _work(self, root: Path) -> None:
        from ..testing.demo_run import advance, demo_services, is_demo

        try:
            for _ in range(20):  # decisions that arrive while it works are picked up by the next round
                run = RunHandle(root)
                services: Any = demo_services(run) if is_demo(run) else build_services(run)
                advance(run, services)
                if not ActionInbox(run.actions_dir).pending():
                    break
        except Exception as exc:  # noqa: BLE001 - report it on the page instead of losing it in a thread
            log.exception("local worker failed for %s", root)
            with self._lock:
                self._errors[root] = f"{type(exc).__name__}: {exc}"
