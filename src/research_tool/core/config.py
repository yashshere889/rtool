"""Typed application configuration, loaded from layered YAML files.

Precedence (later wins): built-in defaults < each config file in order <
explicit overrides (e.g. ``--set llm.backend=mock``). String values may use
``${VAR}`` or ``${VAR:-default}`` to read environment variables.

Every section forbids unknown keys, so a typo in a YAML file fails loudly
instead of being silently ignored.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import ConfigError


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GateMode(StrEnum):
    ASK = "ask"  # stop and wait for a human
    AUTO = "auto"  # approve automatically (selection gates pick option 0)


class PathsConfig(_Section):
    runs_root: Path = Path("~/rtool-runs")
    resources_root: Path | None = None  # shared datasets/models (downloaded once, reused by every run)
    archive_root: Path | None = None  # where `rtool archive` puts finished runs (Barkla: ~/data, backed up)

    def resolved_runs_root(self) -> Path:
        return Path(self.runs_root).expanduser()


class EpisodeConfig(_Section):
    max_minutes: float = Field(default=480, gt=0)
    gate_grace_minutes: float = Field(default=0, ge=0)
    lease_ttl_seconds: float = Field(default=300, gt=0)


class RevisionsConfig(_Section):
    max_rounds: int = Field(default=2, ge=0)


class GatesConfig(_Section):
    default: GateMode = GateMode.ASK
    modes: dict[str, GateMode] = Field(default_factory=dict)

    def mode(self, gate_id: str) -> GateMode:
        return self.modes.get(gate_id, self.default)


class BudgetLimits(_Section):
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_llm_calls: int | None = None
    max_gpu_hours: float | None = None
    max_jobs: int | None = None


class BudgetsConfig(_Section):
    run: BudgetLimits = Field(default_factory=BudgetLimits)
    stage_default: BudgetLimits = Field(default_factory=BudgetLimits)
    stages: dict[str, BudgetLimits] = Field(default_factory=dict)

    def for_stage(self, stage_id: str) -> BudgetLimits:
        return self.stages.get(stage_id, self.stage_default)


class LLMProfile(_Section):
    """How one *kind* of call is made. Agents name profiles, never model ids."""

    reasoning: Literal["off", "on"] = "off"
    temperature: float = 0.0
    top_p: float | None = None
    presence_penalty: float | None = None  # discourage repeating generated text (not the prompt)
    frequency_penalty: float | None = None
    max_tokens: int = Field(default=2048, gt=0)
    thinking_budget: int | None = None
    samples: int = Field(default=1, ge=1)


def _default_profiles() -> dict[str, LLMProfile]:
    # Nemotron model card: reasoning on -> temperature 0.6 / top_p 0.95; reasoning off -> greedy.
    return {
        "extract": LLMProfile(reasoning="off", temperature=0.0, max_tokens=2048),
        "reason": LLMProfile(reasoning="on", temperature=0.6, top_p=0.95, max_tokens=8192, thinking_budget=4096),
        "code": LLMProfile(reasoning="on", temperature=0.6, top_p=0.95, max_tokens=8192, thinking_budget=4096),
        "write": LLMProfile(reasoning="off", temperature=0.0, max_tokens=4096),
        "judge": LLMProfile(
            reasoning="on", temperature=0.6, top_p=0.95, max_tokens=4096, thinking_budget=2048, samples=3
        ),
    }


class LLMServerConfig(_Section):
    """How an episode starts its own vLLM server (``llm.backend: vllm``)."""

    # Apptainer image containing vLLM; "${NEMOTRON_SIF}" works after `module load nemotron/nano-12b-v2`.
    container: str | None = None
    modules: list[str] = Field(default_factory=lambda: ["apptainer/1.3.6"])
    host: str = "127.0.0.1"  # never expose the server beyond the node
    port: int | None = None  # None = pick a free port
    max_model_len: int = Field(default=65536, gt=0)
    max_num_seqs: int = Field(default=64, gt=0)
    gpu_memory_utilization: float = Field(default=0.90, gt=0, le=1)
    reasoning_parser: str | None = "deepseek_r1"  # splits <think> reasoning from the answer
    # Required by Nemotron Nano v2 (model card): remote code + float32 Mamba state for accuracy.
    extra_args: list[str] = Field(default_factory=lambda: ["--trust-remote-code", "--mamba_ssm_cache_dtype", "float32"])
    binds: list[str] = Field(default_factory=list)  # extra Apptainer --bind specs
    hf_home: str | None = None  # Hugging Face cache holding the weights
    # HOME inside the container: keeps vLLM/Triton/FlashInfer caches out of your real home and
    # hides ~/.local packages, which would otherwise shadow the container's own.
    home: str | None = None
    offline: bool = True  # HF_HUB_OFFLINE=1: never download during an episode
    startup_timeout_s: float = Field(default=1200, gt=0)


class LLMConfig(_Section):
    backend: Literal["mock", "vllm", "external"] = "mock"
    model: str = "nvidia/NVIDIA-Nemotron-Nano-12B-v2"  # what the server loads (HF id or local path)
    served_model: str = "nemotron"  # the name requests use
    base_url: str | None = None  # backend=external, e.g. http://127.0.0.1:8000/v1
    api_key_env: str = "RTOOL_LLM_API_KEY"  # backend=external: env var holding the bearer token
    request_timeout_s: float = Field(default=900, gt=0)
    max_retries: int = Field(default=3, ge=0)
    concurrency: int = Field(default=16, ge=1)  # parallel requests for bulk work (continuous batching)
    # How JSON-schema constraints are sent: auto = OpenAI response_format (works across vLLM versions).
    structured_mode: Literal["auto", "response_format", "structured_outputs", "guided_json", "none"] = "auto"
    # Can the server constrain the answer *after* free-form reasoning (needs a reasoning parser)?
    # If not, reasoning + JSON is done in two calls: think first, then format.
    structured_with_reasoning: bool = True
    reasoning_control: Literal["system_flag", "none"] = "system_flag"  # Nemotron: /think or /no_think
    # Spell out each JSON field (name, type, description) at the end of structured prompts, and require
    # every field in the constraint: the constraint alone never shows the model what a field is for.
    describe_schema: bool = True
    server: LLMServerConfig = Field(default_factory=LLMServerConfig)
    profiles: dict[str, LLMProfile] = Field(default_factory=_default_profiles)

    def profile(self, name: str) -> LLMProfile:
        try:
            return self.profiles[name]
        except KeyError:
            raise ConfigError(f"unknown LLM profile {name!r}; configured: {sorted(self.profiles)}") from None


class PartitionSpec(_Section):
    """What the planner and job validator need to know about one Slurm partition."""

    gpu_type: str | None = None  # GRES type name, e.g. "l40s"; None for CPU partitions
    gpu_types: list[str] = Field(default_factory=list)  # every GRES type, if the partition mixes GPUs
    gpus_per_node: int = Field(default=0, ge=0)
    gpu_memory_gb: float | None = None
    max_time: str = "3-00:00:00"
    preemptible: bool = False
    note: str = ""

    def allows_gpu_type(self, gpu_type: str) -> bool:
        return gpu_type in (self.gpu_types or [self.gpu_type])


class GPUChoice(_Section):
    partition: str
    gpu_type: str | None = None


class EpisodeJobConfig(_Section):
    """How `rtool` submits its own episode jobs."""

    partition: str = "gpu-l40s"  # stages that need the LLM run here (vLLM on one GPU)
    gpu_type: str | None = "l40s"
    # Alternatives for LLM episodes, in order of preference: the one Slurm expects to start
    # soonest wins, but an earlier entry is kept unless a later one starts this much sooner.
    gpu_choices: list[GPUChoice] = Field(default_factory=list)
    prefer_within_minutes: float = Field(default=30, ge=0)
    gpus: int = Field(default=1, ge=1)
    cpu_partition: str = "nodes"  # episodes that only apply actions / poll jobs
    time_limit: str = "08:00:00"
    signal_seconds: int = Field(default=600, ge=0)  # warning before the time limit
    modules: list[str] = Field(default_factory=lambda: ["apptainer/1.3.6", "miniforge3/25.3.0-python3.12.10"])
    activate: str | None = None  # e.g. /mnt/fastscratch/users/$USER/rtool/venv/bin/activate
    config_files: list[str] = Field(default_factory=list)  # passed to `rtool episode --config`


class SlurmConfig(_Section):
    backend: Literal["fake", "real"] = "fake"
    # Barkla: jobs must be submitted from scratch or fastscratch. Empty list = no check.
    submit_roots: list[str] = Field(default_factory=list)
    default_time: str = "08:00:00"
    max_jobs_in_queue: int = Field(default=900, gt=0)  # Barkla allows 1000 per user, array tasks included
    partitions: dict[str, PartitionSpec] = Field(default_factory=dict)  # known partitions (empty = no check)
    experiment_partitions: list[str] = Field(default_factory=list)  # where the Coder may run experiments
    command_timeout_s: float = Field(default=60, gt=0)
    episode: EpisodeJobConfig = Field(default_factory=EpisodeJobConfig)


class KnowledgeConfig(_Section):
    """Site settings for the knowledge base. The neural embedder downloads a small model, so it is opt-in."""

    embedder: Literal["hashing", "fastembed"] = "hashing"
    embedder_model: str = "BAAI/bge-small-en-v1.5"
    cache_dir: str | None = None  # where fastembed keeps the model


class ExperimentsConfig(_Section):
    """How the Coder's experiment code runs: smoke tests inside the episode, full runs as Slurm jobs."""

    # Apptainer image with the experiment stack (e.g. "$NEMOTRON_SIF": torch 2.8 + CUDA 12.8).
    # None runs the code with this machine's Python (local development, tests).
    container: str | None = None
    python: str = "python3"  # interpreter inside the container
    modules: list[str] = Field(default_factory=lambda: ["apptainer/1.3.6"])  # loaded in experiment jobs
    binds: list[str] = Field(default_factory=list)
    home: str | None = None  # fresh HOME inside the container (hides ~/.local packages)
    packages: list[str] = Field(default_factory=list)  # importable in the image; shown to the Coder
    smoke_timeout_s: float = Field(default=900, gt=0)  # one smoke run (CPU, tiny data)
    install_timeout_s: float = Field(default=1200, gt=0)  # pip install of extra requirements
    max_parallel_runs: int = Field(default=4, ge=1)  # array throttle: GPUs used at once


class AppConfig(_Section):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    experiments: ExperimentsConfig = Field(default_factory=ExperimentsConfig)
    episode: EpisodeConfig = Field(default_factory=EpisodeConfig)
    revisions: RevisionsConfig = Field(default_factory=RevisionsConfig)
    gates: GatesConfig = Field(default_factory=GatesConfig)
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)


_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def interpolate_env(value: Any, env: Mapping[str, str]) -> Any:
    """Replace ``${VAR}`` / ``${VAR:-default}`` in every string inside ``value``."""
    if isinstance(value, str):

        def substitute(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in env:
                return env[name]
            if default is not None:
                return default
            raise ConfigError(f"environment variable {name} is referenced in config but not set")

        return _ENV_REF.sub(substitute, value)
    if isinstance(value, dict):
        return {k: interpolate_env(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v, env) for v in value]
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (dicts merge, everything else replaces)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def parse_set_overrides(items: Iterable[str]) -> dict[str, Any]:
    """Turn ``["llm.backend=mock", "revisions.max_rounds=3"]`` into a nested dict.

    Values are parsed as YAML, so numbers, booleans and lists work naturally.
    """
    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"--set expects KEY=VALUE, got {item!r}")
        dotted, raw = item.split("=", 1)
        keys = [k for k in dotted.strip().split(".") if k]
        if not keys:
            raise ConfigError(f"--set has an empty key: {item!r}")
        node = result
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ConfigError(f"--set {dotted!r} conflicts with an earlier value")
        node[keys[-1]] = yaml.safe_load(raw)
    return result


def _validate(data: Mapping[str, Any]) -> AppConfig:
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors())
        raise ConfigError(f"invalid configuration: {problems}") from None


def load_config(
    *files: Path,
    overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> AppConfig:
    env = os.environ if env is None else env
    data: dict[str, Any] = AppConfig().model_dump(mode="json")
    for path in files:
        data = deep_merge(data, interpolate_env(load_yaml_mapping(Path(path)), env))
    if overrides:
        data = deep_merge(data, interpolate_env(dict(overrides), env))
    return _validate(data)


def apply_overrides(config: AppConfig, overrides: Mapping[str, Any], env: Mapping[str, str] | None = None) -> AppConfig:
    """Return ``config`` with ``overrides`` merged on top (used for per-invocation settings)."""
    env = os.environ if env is None else env
    return _validate(deep_merge(config.model_dump(mode="json"), interpolate_env(dict(overrides), env)))


def default_config_files(env: Mapping[str, str] | None = None) -> list[Path]:
    """Config files picked up automatically: ``$RTOOL_CONFIG`` (colon-separated) or the user config."""
    env = os.environ if env is None else env
    if env.get("RTOOL_CONFIG"):
        return [Path(p).expanduser() for p in env["RTOOL_CONFIG"].split(os.pathsep) if p]
    user = Path("~/.config/rtool/config.yaml").expanduser()
    return [user] if user.exists() else []
