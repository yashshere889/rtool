"""Turn a JobSpec into an sbatch script that follows Barkla's rules, and check it first.

Rules enforced (Research IT docs, 2025.1):

* ``#!/bin/bash -l`` so ``module`` works inside the job;
* one partition and one node per job;
* GPU jobs request ``--gres=gpu:<type>:<n>`` and *no* CPU/memory (allocated per GPU);
* time limit within the partition's maximum;
* submitted from scratch/fastscratch (compute nodes can only write there);
* never more than ``max_jobs_in_queue`` jobs (Barkla: 1000 per user, array tasks count).

Every script writes *sentinel files* (``<job>.started`` / ``<job>.exit``) so the
outcome is known even where ``sacct`` is unavailable.
"""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Iterable
from pathlib import Path

from ..core.config import SlurmConfig
from ..core.errors import RToolError
from .jobspec import JobSpec, parse_time_limit

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MODULE = re.compile(r"^[A-Za-z0-9_./+-]+$")
_JOB_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


class JobSpecError(RToolError):
    """A job specification breaks a cluster rule or our limits."""


def array_size(spec: str | None) -> int:
    """Number of tasks in an array spec such as '1-10', '1,3,5', '0-15:4' or '1-100%10'."""
    if not spec:
        return 1
    total = 0
    for part in spec.split("%")[0].split(","):
        part = part.strip()
        if "-" in part:
            span, _, step = part.partition(":")
            low, high = (int(x) for x in span.split("-", 1))
            total += len(range(low, high + 1, int(step) if step else 1))
        elif part:
            int(part)
            total += 1
    return total


def _expand(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path))).resolve()


def is_within(path: Path, roots: Iterable[str]) -> bool:
    resolved = Path(path).expanduser().resolve()
    for root in roots:
        root_path = _expand(root)
        if resolved == root_path or root_path in resolved.parents:
            return True
    return False


def validate_job(spec: JobSpec, config: SlurmConfig, *, allowed_partitions: Iterable[str] | None = None) -> list[str]:
    problems: list[str] = []
    if not _JOB_NAME.match(spec.name):
        problems.append(f"job name {spec.name!r} must be 1-100 letters, digits, '.', '_' or '-'")
    allowed = list(allowed_partitions) if allowed_partitions is not None else None
    if allowed is not None and spec.partition not in allowed:
        problems.append(f"partition {spec.partition!r} is not allowed here (allowed: {allowed})")
    partition = config.partitions.get(spec.partition)
    if config.partitions and partition is None:
        problems.append(f"unknown partition {spec.partition!r} (known: {sorted(config.partitions)})")
    if partition is not None:
        if partition.gpus_per_node:
            if spec.gpus is None:
                problems.append(f"{spec.partition} is a GPU partition: request GPUs")
            else:
                if spec.gpus.count > partition.gpus_per_node:
                    problems.append(
                        f"{spec.gpus.count} GPUs requested but {spec.partition} nodes have {partition.gpus_per_node}"
                    )
                if spec.gpus.type and partition.gpu_type and not partition.allows_gpu_type(spec.gpus.type):
                    kinds = "/".join(partition.gpu_types or [partition.gpu_type])
                    problems.append(f"{spec.partition} has {kinds} GPUs, not {spec.gpus.type}")
            if spec.ntasks:
                problems.append("GPU jobs must not request CPU cores on Barkla (they are allocated per GPU)")
        elif spec.gpus is not None:
            problems.append(f"{spec.partition} has no GPUs")
        if spec.time_limit_seconds > parse_time_limit(partition.max_time):
            problems.append(f"time limit {spec.time_limit} exceeds {spec.partition}'s maximum {partition.max_time}")
    if config.submit_roots and not is_within(spec.workdir, config.submit_roots):
        problems.append(
            f"workdir {spec.workdir} is not under {config.submit_roots} "
            "(Barkla: jobs must be submitted from scratch or fastscratch)"
        )
    try:
        tasks = array_size(spec.array)
        if tasks > config.max_jobs_in_queue:
            problems.append(f"array of {tasks} tasks exceeds the queue limit ({config.max_jobs_in_queue})")
    except ValueError:
        problems.append(f"invalid array spec {spec.array!r}")
    problems += [f"invalid environment variable name {k!r}" for k in spec.env if not _ENV_NAME.match(k)]
    problems += [f"invalid module name {m!r}" for m in spec.modules if not _MODULE.match(m)]
    return problems


def sentinel_name(array: bool) -> str:
    return "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" if array else "${SLURM_JOB_ID}"


_NODE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def render_script(spec: JobSpec, *, sentinel_dir: Path, log_dir: Path) -> str:
    """The sbatch script for ``spec`` (call ``validate_job`` first)."""
    array = bool(spec.array)
    stdout = spec.stdout or str(log_dir / (f"{spec.name}-%A_%a.out" if array else f"{spec.name}-%j.out"))
    header = [
        "#!/bin/bash -l",
        f"#SBATCH -J {spec.name}",
        f"#SBATCH -p {spec.partition}",
        "#SBATCH -N 1",
    ]
    if spec.gpus is not None:
        header.append(f"#SBATCH --gres={spec.gpus.gres()}")
    elif spec.ntasks:
        header.append(f"#SBATCH -n {spec.ntasks}")
    header += [
        f"#SBATCH -t {spec.time_limit}",
        f"#SBATCH -D {spec.workdir}",
        f"#SBATCH -o {stdout}",
    ]
    if array:
        header.append(f"#SBATCH -a {spec.array}")
    if spec.dependency:
        header.append(f"#SBATCH --dependency={spec.dependency_type}:{':'.join(spec.dependency)}")
        header.append("#SBATCH --kill-on-invalid-dep=yes")
    nodes = [n for n in spec.exclude if _NODE_NAME.fullmatch(n)]
    if nodes:
        header.append(f"#SBATCH --exclude={','.join(nodes)}")
    if spec.signal:
        header.append(f"#SBATCH --signal={spec.signal}")
    if spec.requeue:
        # A preempted job restarts from the top with the same id; keep the earlier attempt's log.
        header += ["#SBATCH --requeue", "#SBATCH --open-mode=append"]
    else:
        header.append("#SBATCH --no-requeue")

    command = spec.command if isinstance(spec.command, str) else shlex.join(spec.command)
    if spec.signal and spec.signal.startswith("B:"):
        # The warning signal goes to this batch shell; pass it on to the real work. TERM
        # (preemption, scancel) is passed on too, and the shell waits so the work can stop cleanly.
        sig = spec.signal[2:].split("@", 1)[0]
        command = "\n".join(
            [
                f"( {command} ) &",
                "child=$!",
                f"trap 'kill -{sig} \"$child\" 2>/dev/null' {sig}",
                "trap 'kill -TERM \"$child\" 2>/dev/null' TERM",
                'wait "$child"; status=$?',
                'while [ "$status" -gt 128 ] && kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done',
                'exit "$status"',
            ]
        )

    body = [
        "",
        "# Written by research_tool. Sentinel files record the outcome even without sacct.",
        f"RTOOL_SENTINEL={shlex.quote(str(sentinel_dir))}/{sentinel_name(array)}",
        'mkdir -p "$(dirname "$RTOOL_SENTINEL")"',
        'rm -f "$RTOOL_SENTINEL.exit"  # left by an earlier attempt if the job was requeued',
        'date +%Y-%m-%dT%H:%M:%S%z > "$RTOOL_SENTINEL.started"',
        "trap 'echo $? > \"$RTOOL_SENTINEL.exit\"' EXIT",
        "module purge",
        *[f"module load {m}" for m in spec.modules],
        *[f"export {k}={shlex.quote(v)}" for k, v in spec.env.items()],
        "",
        command,
        "",
    ]
    return "\n".join(header + body)
