"""SubprocessSlurm against a scripted fake of sbatch / squeue / sacct / scancel."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from research_tool.core.config import PartitionSpec, SlurmConfig
from research_tool.hpc.jobspec import GPURequest, JobSpec
from research_tool.hpc.sbatch import JobSpecError
from research_tool.hpc.slurm import JobState
from research_tool.hpc.slurm_client import SlurmCommandError, SubprocessSlurm, parse_sacct_state

CONFIG = SlurmConfig(
    backend="real",
    partitions={"gpu-l40s": PartitionSpec(gpu_type="l40s", gpus_per_node=2), "nodes": PartitionSpec()},
    max_jobs_in_queue=5,
)


class FakeCluster:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.queue: dict[str, str] = {}  # squeue view: job id -> state
        self.accounting: list[str] = []  # sacct rows "JobID|State|ExitCode"
        self.sacct_works = True
        self.scripts: dict[str, str] = {}
        self.next_id = 5000
        self.tested: list[str] = []
        self.test_only_answer = (
            "sbatch: Job 77 to start at 2026-10-08T15:30:02 using 24 processors on nodes gpu47 in partition gpu-l40s\n"
        )

    def __call__(self, args: list[str], cwd: Path | None, timeout: float) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        done = lambda out="", code=0, err="": subprocess.CompletedProcess(args, code, out, err)  # noqa: E731
        if args[0] == "sbatch" and "--test-only" in args:
            self.tested.append(Path(args[-1]).read_text())
            return done(err=self.test_only_answer)
        if args[0] == "sbatch":
            job_id = str(self.next_id)
            self.next_id += 1
            self.queue[job_id] = "PENDING"
            self.scripts[job_id] = Path(args[-1]).read_text()
            return done(f"{job_id};barkla\n")
        if args[0] == "squeue":
            fmt = args[args.index("-o") + 1]
            rows = [job if fmt == "%i" else f"{job}|{state}|None" for job, state in self.queue.items()]
            return done("\n".join(rows) + "\n")
        if args[0] == "sacct":
            if not self.sacct_works:
                return done(code=1, err="Slurm accounting storage is disabled")
            return done("\n".join(self.accounting) + "\n")
        if args[0] == "scancel":
            self.queue.pop(args[1], None)
            return done()
        raise FileNotFoundError(args[0])


@pytest.fixture
def cluster() -> FakeCluster:
    return FakeCluster()


def client(tmp_path: Path, cluster: FakeCluster, now: float = 1_000_000.0) -> SubprocessSlurm:
    return SubprocessSlurm(CONFIG, state_dir=tmp_path / "jobs", runner=cluster, user="me", clock=lambda: now)


def job(tmp_path: Path, **kw: object) -> JobSpec:
    base = {
        "name": "exp",
        "partition": "gpu-l40s",
        "command": "python run.py",
        "workdir": tmp_path,
        "gpus": GPURequest(type="l40s", count=1),
    }
    return JobSpec(**(base | kw))  # type: ignore[arg-type]


def test_submit_renders_script_and_records_job(tmp_path: Path, cluster: FakeCluster) -> None:
    slurm = client(tmp_path, cluster)
    job_id = slurm.submit(job(tmp_path))
    assert job_id == "5000"
    assert "#SBATCH --gres=gpu:l40s:1" in cluster.scripts[job_id]
    assert (tmp_path / ".rtool" / "logs").is_dir()  # Slurm needs the -o directory to exist
    assert (tmp_path / "jobs" / "registry" / "5000.json").exists()


def test_invalid_jobs_and_queue_limit_are_refused(tmp_path: Path, cluster: FakeCluster) -> None:
    slurm = client(tmp_path, cluster)
    with pytest.raises(JobSpecError, match="nodes have 2"):
        slurm.submit(job(tmp_path, gpus=GPURequest(type="l40s", count=3)))
    for i in range(4):
        cluster.queue[f"9{i}"] = "RUNNING"
    with pytest.raises(JobSpecError, match="would pass the limit of 5"):
        slurm.submit(job(tmp_path, array="1-2"))
    assert not any(call[0] == "sbatch" for call in cluster.calls)


def test_status_from_squeue_then_sacct(tmp_path: Path, cluster: FakeCluster) -> None:
    slurm = client(tmp_path, cluster)
    a = slurm.submit(job(tmp_path))
    b = slurm.submit(job(tmp_path, array="1-3"))
    cluster.queue[a] = "RUNNING"
    assert slurm.status(a).state is JobState.RUNNING
    del cluster.queue[a], cluster.queue[b]
    cluster.accounting = [
        f"{a}|COMPLETED|0:0",
        f"{b}_1|COMPLETED|0:0",
        f"{b}_2|TIMEOUT|0:1",
        f"{b}_3|CANCELLED by 123|0:0",
    ]
    statuses = slurm.status_many([a, b])
    assert statuses[a].state is JobState.COMPLETED and statuses[a].exit_code == 0
    assert statuses[b].state is JobState.TIMEOUT  # worst array task wins


def test_sentinel_fallback_when_sacct_is_unavailable(tmp_path: Path, cluster: FakeCluster) -> None:
    cluster.sacct_works = False
    later = client(tmp_path, cluster, now=1_000_000.0 + 3600)
    ok, bad, killed, vanished = (client(tmp_path, cluster).submit(job(tmp_path)) for _ in range(4))
    cluster.queue.clear()
    sentinels = tmp_path / ".rtool" / "sentinels"
    for job_id, code in ((ok, "0"), (bad, "2")):
        (sentinels / f"{job_id}.started").write_text("t")
        (sentinels / f"{job_id}.exit").write_text(code)
    (sentinels / f"{killed}.started").write_text("t")
    statuses = later.status_many([ok, bad, killed, vanished])
    assert statuses[ok].state is JobState.COMPLETED
    assert statuses[bad].state is JobState.FAILED and statuses[bad].exit_code == 2
    assert statuses[killed].state is JobState.FAILED and "without an exit record" in statuses[killed].reason
    assert statuses[vanished].state is JobState.CANCELLED
    assert later.sacct_available is False
    # a job submitted moments ago that squeue doesn't show yet counts as pending
    assert client(tmp_path, cluster).status(vanished).state is JobState.PENDING


def test_missing_commands_give_a_clear_error(tmp_path: Path) -> None:
    def no_slurm(args: list[str], cwd: Path | None, timeout: float) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(args[0])

    slurm = SubprocessSlurm(CONFIG, state_dir=tmp_path, runner=no_slurm, user="me")
    with pytest.raises(SlurmCommandError, match="squeue not found"):
        slurm.status("1")


def test_parse_sacct_state() -> None:
    assert parse_sacct_state("CANCELLED by 1234") is JobState.CANCELLED
    assert parse_sacct_state("COMPLETED") is JobState.COMPLETED
    assert parse_sacct_state("RUNNING") is JobState.RUNNING
    assert parse_sacct_state("SOMETHING_NEW") is JobState.UNKNOWN
    assert parse_sacct_state("REQUEUED") is JobState.PENDING


def test_estimate_start_asks_slurm_without_submitting(tmp_path: Path, cluster: FakeCluster) -> None:
    slurm = client(tmp_path, cluster)
    assert slurm.estimate_start(job(tmp_path)) == datetime(2026, 10, 8, 15, 30, 2).timestamp()
    assert "#SBATCH --gres=gpu:l40s:1" in cluster.tested[0]
    assert not cluster.queue and not list((tmp_path / ".rtool" / "scripts").iterdir())  # nothing left behind
    cluster.test_only_answer = (
        "sbatch: error: Batch job submission failed: Requested node configuration is not available"
    )
    assert slurm.estimate_start(job(tmp_path)) is None
    with pytest.raises(JobSpecError):
        slurm.estimate_start(job(tmp_path, gpus=GPURequest(type="l40s", count=3)))


def test_preempted_jobs(tmp_path: Path, cluster: FakeCluster) -> None:
    slurm = client(tmp_path, cluster)
    job_id = slurm.submit(job(tmp_path))
    cluster.queue[job_id] = "PREEMPTED"  # still queued: about to be requeued
    assert slurm.status(job_id).state is JobState.PENDING
    del cluster.queue[job_id]
    cluster.accounting = [f"{job_id}|PREEMPTED|0:0"]  # left the queue: over, and not a success
    state = slurm.status(job_id).state
    assert state is JobState.PREEMPTED and state.is_terminal and not state.succeeded
