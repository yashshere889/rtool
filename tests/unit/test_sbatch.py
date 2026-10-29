from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from research_tool.core.config import PartitionSpec, SlurmConfig
from research_tool.hpc.jobspec import GPURequest, JobSpec
from research_tool.hpc.sbatch import array_size, is_within, render_script, validate_job

BARKLA = SlurmConfig(
    partitions={
        "gpu-l40s": PartitionSpec(gpu_type="l40s", gpus_per_node=2, max_time="3-00:00:00"),
        "nodes": PartitionSpec(max_time="3-00:00:00"),
    },
    max_jobs_in_queue=900,
)


def gpu_job(workdir: Path, **kw: object) -> JobSpec:
    base = {
        "name": "train",
        "partition": "gpu-l40s",
        "command": ["python", "train.py", "--epochs", "3"],
        "workdir": workdir,
        "time_limit": "02:00:00",
        "gpus": GPURequest(type="l40s", count=1),
    }
    return JobSpec(**(base | kw))  # type: ignore[arg-type]


@pytest.mark.parametrize(("text", "n"), [(None, 1), ("1-10", 10), ("1,3,5", 3), ("0-15:4", 4), ("1-100%10", 100)])
def test_array_size(text: str | None, n: int) -> None:
    assert array_size(text) == n


def test_is_within(tmp_path: Path) -> None:
    (tmp_path / "scratch" / "run").mkdir(parents=True)
    assert is_within(tmp_path / "scratch" / "run", [str(tmp_path / "scratch")])
    assert not is_within(tmp_path / "home", [str(tmp_path / "scratch")])
    assert not is_within(tmp_path / "scratchy", [str(tmp_path / "scratch")])  # prefix is not containment


def test_valid_gpu_job_passes(tmp_path: Path) -> None:
    assert validate_job(gpu_job(tmp_path), BARKLA) == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"partition": "gpu-a100-cs"}, "unknown partition"),
        ({"gpus": None}, "GPU partition: request GPUs"),
        ({"gpus": GPURequest(type="l40s", count=4)}, "nodes have 2"),
        ({"gpus": GPURequest(type="h100", count=1)}, "has l40s GPUs"),
        ({"ntasks": 8}, "must not request CPU cores"),
        ({"time_limit": "4-00:00:00"}, "exceeds"),
        ({"partition": "nodes"}, "nodes has no GPUs"),
        ({"array": "1-1000"}, "exceeds the queue limit"),
        ({"env": {"BAD-NAME": "x"}}, "invalid environment variable"),
        ({"modules": ["cuda; rm -rf /"]}, "invalid module name"),
        ({"name": "has space"}, "job name"),
    ],
)
def test_rule_violations(tmp_path: Path, changes: dict, message: str) -> None:
    problems = validate_job(gpu_job(tmp_path, **changes), BARKLA)
    assert any(message in p for p in problems), problems


def test_partitions_with_mixed_gpus(tmp_path: Path) -> None:
    mixed = PartitionSpec(gpu_type="a40", gpu_types=["a40", "a100"], gpus_per_node=2)
    cfg = SlurmConfig(partitions={"gpu-a-lowsmall": mixed})
    a100 = gpu_job(tmp_path, partition="gpu-a-lowsmall", gpus=GPURequest(type="a100"))
    assert validate_job(a100, cfg) == []
    v100 = gpu_job(tmp_path, partition="gpu-a-lowsmall", gpus=GPURequest(type="v100"))
    assert any("has a40/a100 GPUs, not v100" in p for p in validate_job(v100, cfg))


def test_submit_roots_and_allow_list(tmp_path: Path) -> None:
    cfg = BARKLA.model_copy(update={"submit_roots": [str(tmp_path / "scratch")]})
    assert any("must be submitted from scratch" in p for p in validate_job(gpu_job(tmp_path / "home"), cfg))
    assert any("not allowed here" in p for p in validate_job(gpu_job(tmp_path), BARKLA, allowed_partitions=["nodes"]))


def test_rendered_script_follows_barkla_rules(tmp_path: Path) -> None:
    spec = gpu_job(
        tmp_path,
        dependency=["11", "12"],
        modules=["apptainer/1.3.6"],
        env={"HF_HOME": "/mnt/scratch/u/hf cache"},
        signal="B:USR1@600",
    )
    script = render_script(spec, sentinel_dir=tmp_path / "s", log_dir=tmp_path / "l")
    lines = script.splitlines()
    assert lines[0] == "#!/bin/bash -l"
    for expected in (
        "#SBATCH -p gpu-l40s",
        "#SBATCH -N 1",
        "#SBATCH --gres=gpu:l40s:1",
        "#SBATCH -t 02:00:00",
        "#SBATCH --dependency=afterany:11:12",
        "#SBATCH --kill-on-invalid-dep=yes",
        "#SBATCH --signal=B:USR1@600",
        "#SBATCH --no-requeue",
        "module purge",
        "module load apptainer/1.3.6",
        "export HF_HOME='/mnt/scratch/u/hf cache'",
    ):
        assert expected in lines, expected
    assert not any(line.startswith("#SBATCH -n") or "--mem" in line for line in lines)  # GPU job: no CPU/mem
    assert "( python train.py --epochs 3 ) &" in script


def _run_script(script: Path, env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(["bash", str(script)], env=os.environ | env, cwd=script.parent)


def test_script_records_exit_code_sentinel(tmp_path: Path) -> None:
    spec = JobSpec(name="t", partition="nodes", command="echo working; exit 3", workdir=tmp_path, ntasks=1)
    script = tmp_path / "job.sh"
    script.write_text(render_script(spec, sentinel_dir=tmp_path / "sent", log_dir=tmp_path / "logs"))
    proc = _run_script(script, {"SLURM_JOB_ID": "42"})
    assert proc.wait(timeout=30) == 3
    assert (tmp_path / "sent" / "42.started").exists()
    assert (tmp_path / "sent" / "42.exit").read_text().split()[0] == "3"


def test_warning_signal_is_forwarded_to_the_work(tmp_path: Path) -> None:
    worker = (
        f'{shlex.quote(sys.executable)} -c "import signal, sys, time; '
        "signal.signal(signal.SIGUSR1, lambda *a: (open('got-signal', 'w').write('1'), sys.exit(7))); "
        'time.sleep(20)"'
    )
    spec = JobSpec(name="t", partition="nodes", command=worker, workdir=tmp_path, ntasks=1, signal="B:USR1@600")
    script = tmp_path / "job.sh"
    script.write_text(render_script(spec, sentinel_dir=tmp_path / "sent", log_dir=tmp_path / "logs"))
    proc = _run_script(script, {"SLURM_JOB_ID": "43"})
    time.sleep(1.0)  # let the worker install its handler
    proc.send_signal(signal.SIGUSR1)  # what Slurm does with --signal=B:USR1@600
    assert proc.wait(timeout=30) == 7
    assert (tmp_path / "got-signal").exists()
    assert (tmp_path / "sent" / "43.exit").read_text().split()[0] == "7"


def test_requeued_job_script(tmp_path: Path) -> None:
    spec = gpu_job(tmp_path, requeue=True, signal="B:USR1@600")
    lines = render_script(spec, sentinel_dir=tmp_path / "s", log_dir=tmp_path / "l").splitlines()
    assert "#SBATCH --requeue" in lines and "#SBATCH --open-mode=append" in lines
    assert "#SBATCH --no-requeue" not in lines
    assert any(line.startswith('rm -f "$RTOOL_SENTINEL.exit"') for line in lines)  # stale record of the last attempt


def test_termination_waits_for_the_work_to_clean_up(tmp_path: Path) -> None:
    # Preemption sends TERM; the batch shell must not exit before the work has stopped cleanly.
    worker = (
        f'{shlex.quote(sys.executable)} -c "import signal, sys, time; '
        "cleanup = lambda *a: (time.sleep(0.5), open('cleaned-up', 'w').write('1'), sys.exit(143)); "
        'signal.signal(signal.SIGTERM, cleanup); time.sleep(20)"'
    )
    spec = JobSpec(name="t", partition="nodes", command=worker, workdir=tmp_path, ntasks=1, signal="B:USR1@600")
    script = tmp_path / "job.sh"
    script.write_text(render_script(spec, sentinel_dir=tmp_path / "sent", log_dir=tmp_path / "logs"))
    proc = _run_script(script, {"SLURM_JOB_ID": "44"})
    time.sleep(1.0)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 143
    assert (tmp_path / "cleaned-up").exists()  # the shell waited for the cleanup


def test_bad_nodes_are_excluded(tmp_path: Path) -> None:
    spec = gpu_job(tmp_path).model_copy(update={"exclude": ["gpu45", "gpu46", "bad;rm -rf"]})
    lines = render_script(spec, sentinel_dir=tmp_path / "s", log_dir=tmp_path / "l").splitlines()
    assert "#SBATCH --exclude=gpu45,gpu46" in lines  # anything that is not a node name is left out
    assert not any(
        "--exclude" in line
        for line in render_script(gpu_job(tmp_path), sentinel_dir=tmp_path / "s", log_dir=tmp_path / "l").splitlines()
    )
