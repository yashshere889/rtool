"""External services handed to the engine (LLM, Slurm, clock).

Keeping them in one injectable container is what lets the same engine run
against a mock LLM and fake Slurm on a laptop, and against vLLM and real Slurm
on Barkla.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .config import AppConfig
from .errors import ConfigError

if TYPE_CHECKING:
    from ..hpc.slurm import SlurmClient
    from ..llm.client import LLMClient
    from .run import RunHandle


@dataclass
class Services:
    llm: LLMClient | None = None
    slurm: SlurmClient | None = None
    clock: Callable[[], float] = field(default=time.time)
    # Named tool factories agents may use instead of their defaults (fakes in tests, site-specific clients).
    tools: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMEndpoint:
    """A running OpenAI-compatible server (e.g. the vLLM server an episode started)."""

    url: str
    api_key: str | None = None


def build_llm(config: AppConfig, endpoint: LLMEndpoint | None = None) -> LLMClient:
    from ..llm.client import LLMClient

    settings = config.llm
    if endpoint is None:  # an explicitly passed endpoint always wins
        if settings.backend == "mock":
            from ..llm.mock_backend import MockLLM

            return LLMClient(MockLLM(), settings)
        if settings.backend == "external":
            if not settings.base_url:
                raise ConfigError("llm.backend=external needs llm.base_url (e.g. http://127.0.0.1:8000/v1)")
            endpoint = LLMEndpoint(url=settings.base_url, api_key=os.environ.get(settings.api_key_env))
        else:
            raise ConfigError(
                "llm.backend=vllm needs a running server: run inside an episode job, "
                "or use `rtool run --serve-llm` on a GPU node"
            )
    from ..llm.vllm_backend import VLLMBackend

    backend = VLLMBackend(
        endpoint.url,
        model=settings.served_model,
        api_key=endpoint.api_key,
        timeout_s=settings.request_timeout_s,
        max_retries=settings.max_retries,
        structured_mode=settings.structured_mode,
        reasoning_control=settings.reasoning_control,
    )
    return LLMClient(backend, settings)


def build_slurm(run: RunHandle) -> SlurmClient:
    if run.config.slurm.backend == "fake":
        from ..hpc.fake_slurm import FakeSlurm

        return FakeSlurm(run.jobs_dir / "fake_slurm.json")
    from ..hpc.slurm_client import SubprocessSlurm

    return SubprocessSlurm(run.config.slurm, state_dir=run.jobs_dir)


def build_services(run: RunHandle, *, llm_endpoint: LLMEndpoint | None = None) -> Services:
    """Construct services from the run's (possibly overridden) config."""
    return Services(llm=build_llm(run.config, llm_endpoint), slurm=build_slurm(run))
