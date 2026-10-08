"""Where and how experiment code runs.

On Barkla the code runs inside the experiment image (``experiments.container``,
e.g. the Nemotron image: torch 2.8 + CUDA 12.8) with a fresh ``HOME`` and no
``~/.local`` packages. Extra requirements go into a small venv *layered on* the
image (``--system-site-packages``), so pip only installs what the image lacks
and never a second copy of torch. Without a container (local development,
tests) this machine's Python is used.

Smoke tests run here, inside the episode, on CPU. Full runs are Slurm array
jobs built by :meth:`ExperimentEnv.array_job`.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ...core.config import ExperimentsConfig
from ...core.errors import RToolError
from ...core.fs import atomic_write_text
from ...hpc.jobspec import GPURequest, JobSpec
from ...schemas.plan import ComputeRequest

VENV = ".venv"


def expand(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def _import_name(requirement: str) -> str:
    """Best guess at the module a pip requirement provides ('scikit-learn>=1' -> 'sklearn')."""
    from .playbook import IMPORT_TO_PIP

    name = re.split(r"[\[<>=!~]", requirement, maxsplit=1)[0].strip()
    reverse = {pip.lower(): module for module, pip in IMPORT_TO_PIP.items()}
    return reverse.get(name.lower(), name.replace("-", "_"))


def shell_word(arg: str) -> str:
    """Quote for bash, but leave a leading $VAR to be expanded by the job (after `module load`)."""
    if arg.startswith("$") and " " not in arg:
        return f'"{arg}"'
    return shlex.quote(arg)


@dataclass
class CommandResult:
    exit_code: int | None  # None: killed after the timeout
    seconds: float
    log_path: Path
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def tail(self, lines: int = 60) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return ""
        return "\n".join(text.splitlines()[-lines:])


class ExperimentEnv:
    def __init__(self, config: ExperimentsConfig, code_dir: Path, *, binds: list[str] | None = None) -> None:
        self.config = config
        self.code_dir = Path(code_dir)
        self.extra_binds = list(binds or [])

    # ------------------------------------------------------------- commands
    @property
    def containerized(self) -> bool:
        return bool(self.config.container)

    def interpreter(self) -> str:
        if (self.code_dir / VENV / "bin" / "python").exists():
            return str(self.code_dir / VENV / "bin" / "python")
        return self.config.python if self.containerized else sys.executable

    def python(self, args: list[str], *, gpu: bool, for_job: bool = False) -> list[str]:
        """Command running ``python <args>`` in the experiment environment.

        ``for_job`` keeps ``$VARS`` (such as ``$NEMOTRON_SIF``) for the job's shell to expand
        after ``module load``; otherwise they are expanded now, from this process's environment.
        """
        if not self.containerized:
            return [self.interpreter(), *args]
        assert self.config.container is not None
        container = self.config.container if for_job else expand(self.config.container)
        if not for_job and "$" in container:
            raise RToolError(
                f"experiments.container {self.config.container!r} is not set here "
                "(load the module that defines it, e.g. nemotron/nano-12b-v2)"
            )
        command = ["apptainer", "exec"]
        if gpu:
            command.append("--nv")
        if self.config.home:
            home = expand(self.config.home)
            with contextlib.suppress(OSError):  # apptainer needs it to exist (jobs reuse it)
                Path(home).mkdir(parents=True, exist_ok=True)
            command += ["--home", home]
        for bind in [*self.config.binds, *self.extra_binds]:
            command += ["--bind", expand(bind)]
        return [*command, container, self.interpreter(), *args]

    def variables(self, *, gpu: bool) -> dict[str, str]:
        """Environment for the experiment (also passed into the container, beating the image's own)."""
        inner = {
            "PYTHONNOUSERSITE": "1",  # never import ~/.local packages
            "PYTHONUNBUFFERED": "1",  # logs appear as they are written
            "PIP_CONFIG_FILE": "/dev/null",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "MPLBACKEND": "Agg",
            "HF_HUB_OFFLINE": "1",  # data and models come from the fetched resources, not the internet
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        if not gpu:
            inner["CUDA_VISIBLE_DEVICES"] = ""
        if self.containerized:
            inner.update({f"APPTAINERENV_{key}": value for key, value in list(inner.items())})
        return inner

    def environment(self, *, gpu: bool) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
        env.update(self.variables(gpu=gpu))
        return env

    # ------------------------------------------------------------ local runs
    def run(
        self, args: list[str], *, log_path: Path, timeout_s: float, gpu: bool = False, raw: bool = False
    ) -> CommandResult:
        """Run ``python <args>`` (or ``args`` itself when ``raw``) in the code directory, logging to a file."""
        command = list(args) if raw else self.python(args, gpu=gpu)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with open(log_path, "wb") as log:
            log.write(f"$ {shlex.join(command)}\n".encode())
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=self.code_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=self.environment(gpu=gpu),
                start_new_session=True,  # a timeout kills the whole process group
            )
            try:
                code: int | None = process.wait(timeout=timeout_s)
                timed_out = False
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code, timed_out = None, True
        return CommandResult(code, round(time.monotonic() - started, 2), log_path, timed_out)

    def install(self, requirements: list[str], *, log_path: Path) -> CommandResult | None:
        """Install extra packages into a venv layered on the image (nothing to do without requirements).

        Without a container nothing is ever downloaded: this machine's Python is used as it is,
        and a missing package is reported (it then counts as a failure to repair).
        """
        if not requirements:
            return None
        if not self.containerized:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                log_path,
                f"not installing {requirements}: without experiments.container, experiments use this Python "
                "as it is and nothing is downloaded. Install the packages yourself, or run on the cluster.\n",
            )
            missing = [r for r in requirements if importlib.util.find_spec(_import_name(r)) is None]
            return CommandResult(1 if missing else 0, 0.0, log_path)
        venv = self.code_dir / VENV
        if not (venv / "bin" / "python").exists():
            base = self.python(["-m", "venv", "--system-site-packages", str(venv)], gpu=False)
            created = self.run(base, log_path=log_path.with_suffix(".venv.log"), timeout_s=300, raw=True)
            if not created.ok:
                return created
        (self.code_dir / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
        return self.run(
            ["-m", "pip", "install", "--quiet", "-r", "requirements.txt"],
            log_path=log_path,
            timeout_s=self.config.install_timeout_s,
        )

    # -------------------------------------------------------------- full runs
    def array_job(
        self,
        *,
        name: str,
        workdir: Path,
        batch_file: str,
        tasks: int,
        compute: ComputeRequest,
        modules: list[str] | None = None,
    ) -> JobSpec:
        """One Slurm array task per planned run; at most ``max_parallel_runs`` at once."""
        gpus = GPURequest(type=compute.gpu_type, count=compute.gpus_per_run) if compute.gpus_per_run else None
        python = self.python(
            ["launch.py", "--batch", batch_file, "--task", "$SLURM_ARRAY_TASK_ID"], gpu=bool(gpus), for_job=True
        )
        command = f"cd {shlex.quote(str(self.code_dir))} && " + " ".join(shell_word(a) for a in python)
        throttle = min(self.config.max_parallel_runs, tasks)
        return JobSpec(
            name=name,
            partition=compute.partition,
            command=command,
            workdir=workdir,
            time_limit=compute.time_limit,
            gpus=gpus,
            ntasks=None if gpus else 1,
            array=f"0-{tasks - 1}%{throttle}",
            env=self.variables(gpu=bool(gpus)),
            modules=list(modules if modules is not None else self.config.modules),
            requeue=False,
            metadata={"kind": "experiment", "batch": batch_file},
        )
