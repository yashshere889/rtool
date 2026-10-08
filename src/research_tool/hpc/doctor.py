"""`rtool doctor`: check that this machine can run the pipeline with the current config."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx

from .. import __version__
from ..core.config import AppConfig
from .sbatch import is_within
from .storage import QuotaError, lustre_quota

ENDPOINTS = {
    "OpenAlex": "https://api.openalex.org/works?per-page=1",
    "Semantic Scholar": "https://api.semanticscholar.org/graph/v1/paper/search?query=test&limit=1",
    "arXiv": "https://export.arxiv.org/api/query?search_query=all:test&max_results=1",
    "Crossref": "https://api.crossref.org/works?rows=1",
    "Hugging Face": "https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-Nano-12B-v2",
}


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail | skip
    detail: str


def _sqlite_fts5() -> Check:
    try:
        sqlite3.connect(":memory:").execute("create virtual table t using fts5(x)")
        return Check("sqlite FTS5", "ok", f"sqlite {sqlite3.sqlite_version}")
    except sqlite3.OperationalError as exc:
        return Check("sqlite FTS5", "fail", f"full-text search unavailable: {exc}")


def _slurm(cfg: AppConfig) -> list[Check]:
    if cfg.slurm.backend != "real":
        return [Check("slurm", "skip", "slurm.backend is 'fake'")]
    checks = []
    missing = [c for c in ("sbatch", "squeue", "scancel", "sinfo") if shutil.which(c) is None]
    checks.append(
        Check("slurm commands", "fail" if missing else "ok", f"missing: {missing}" if missing else "all found")
    )
    if shutil.which("sacct") is None:
        checks.append(Check("sacct", "warn", "not found; job outcomes will come from sentinel files"))
    if shutil.which("sinfo"):
        out = subprocess.run(["sinfo", "-h", "-o", "%P"], capture_output=True, text=True, timeout=60, check=False)
        known = {p.strip().rstrip("*") for p in out.stdout.split()}
        episode = cfg.slurm.episode
        wanted = set(cfg.slurm.experiment_partitions) | {episode.partition, episode.cpu_partition}
        wanted |= {choice.partition for choice in episode.gpu_choices}
        absent = sorted(p for p in wanted if p not in known)
        checks.append(
            Check(
                "partitions",
                "fail" if absent else "ok",
                f"not on this cluster: {absent}" if absent else ", ".join(sorted(wanted)),
            )
        )
    runs_root = cfg.paths.resolved_runs_root()
    if cfg.slurm.submit_roots and not is_within(runs_root, cfg.slurm.submit_roots):
        checks.append(Check("runs_root", "fail", f"{runs_root} is not under {cfg.slurm.submit_roots}"))
    return checks


ModuleEnv = Callable[[list[str]], dict[str, str] | None]
_VAR = re.compile(r"\$(\w+)|\$\{(\w+)\}")


def expand_with(text: str, env: Mapping[str, str]) -> str:
    """Like os.path.expandvars, but against the given environment."""
    return _VAR.sub(lambda m: env.get(m.group(1) or m.group(2), m.group(0)), text)


def module_environment(modules: list[str]) -> dict[str, str] | None:
    """The environment after `module load <modules>`, read from a throwaway login shell (None if that fails)."""
    if not modules or shutil.which("bash") is None:
        return None
    script = "module load " + " ".join(shlex.quote(m) for m in modules) + " >/dev/null 2>&1 && env -0"
    try:
        proc = subprocess.run(["bash", "-lc", script], capture_output=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    items = proc.stdout.decode(errors="replace").split("\0")
    return dict(item.split("=", 1) for item in items if "=" in item)


def _llm(cfg: AppConfig, module_env: ModuleEnv = module_environment) -> list[Check]:
    if cfg.llm.backend == "mock":
        return [Check("llm", "warn", "llm.backend is 'mock' (no real model)")]
    if cfg.llm.backend == "external":
        return [Check("llm", "ok" if cfg.llm.base_url else "fail", f"external server {cfg.llm.base_url}")]
    server = cfg.llm.server
    env: Mapping[str, str] = os.environ
    container = expand_with(server.container or "", env)
    via = ""
    if ("$" in container or shutil.which("apptainer") is None) and server.modules:
        # The episode job loads llm.server.modules before starting vLLM; do the same here.
        loaded = module_env(server.modules)
        if loaded is not None:
            env, via = loaded, f" (after module load {' '.join(server.modules)})"
            container = expand_with(server.container or "", env)
    checks = []
    if not container or "$" in container:
        checks.append(
            Check(
                "vLLM image",
                "fail",
                f"llm.server.container={server.container!r} does not "
                "expand (load the module that defines it, e.g. nemotron/nano-12b-v2)",
            )
        )
    elif not Path(container).exists():
        checks.append(Check("vLLM image", "fail", f"{container} does not exist"))
    else:
        checks.append(Check("vLLM image", "ok", container + via))
    if shutil.which("apptainer", path=env.get("PATH")) is None:
        checks.append(Check("apptainer", "fail", "not on PATH (module load apptainer/1.3.6)"))
    if server.hf_home:
        cache = Path(os.path.expandvars(os.path.expanduser(server.hf_home))) / "hub"
        model_dir = cache / ("models--" + cfg.llm.model.replace("/", "--"))
        status = "ok" if model_dir.exists() else ("fail" if server.offline else "warn")
        checks.append(
            Check(
                "model weights",
                status,
                str(model_dir)
                if model_dir.exists()
                else f"{cfg.llm.model} not in {cache} (download it once with scripts/fetch_model.sh)",
            )
        )
    return checks


def _quotas() -> list[Check]:
    checks = []
    for mount in ("/mnt/scratch", "/mnt/fastscratch"):
        if not Path(mount).exists():
            continue
        try:
            quota = lustre_quota(mount)
        except QuotaError as exc:
            checks.append(Check(f"quota {mount}", "warn", str(exc)))
            continue
        bytes_frac, files_frac = quota.headroom()
        worst = max(f for f in (bytes_frac, files_frac, 0.0) if f is not None)
        status = "fail" if worst > 0.95 else "warn" if worst > 0.8 else "ok"
        detail = f"space {bytes_frac or 0:.0%}, files {files_frac or 0:.0%} of quota"
        checks.append(Check(f"quota {mount}", status, detail))
    return checks


def _internet(get: Callable[[str], int] | None = None) -> list[Check]:
    def default_get(url: str) -> int:
        return httpx.get(url, timeout=10, follow_redirects=True).status_code

    fetch = get or default_get
    checks = []
    for name, url in ENDPOINTS.items():
        try:
            code = fetch(url)
            checks.append(Check(f"reach {name}", "ok" if code < 500 else "warn", f"HTTP {code}"))
        except httpx.HTTPError as exc:
            checks.append(Check(f"reach {name}", "fail", f"{type(exc).__name__}: {exc}"[:150]))
    return checks


def run_doctor(cfg: AppConfig, *, network: bool = True, module_env: ModuleEnv = module_environment) -> list[Check]:
    checks = [Check("rtool", "ok", f"{__version__} on Python {sys.version.split()[0]}"), _sqlite_fts5()]
    checks += _slurm(cfg)
    checks += _llm(cfg, module_env)
    checks += _quotas()
    if network:
        checks += _internet()
    return checks
