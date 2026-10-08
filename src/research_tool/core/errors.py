"""Exception hierarchy shared across the framework."""

from __future__ import annotations

from typing import Any


class RToolError(Exception):
    """Base class for all framework errors."""


class ConfigError(RToolError):
    """Invalid or unreadable configuration."""


class PipelineError(RToolError):
    """Pipeline specification is inconsistent (unknown agent, cycle, missing producer, ...)."""


class ArtifactError(RToolError):
    """Artifact missing, unregistered, or failing validation."""


class RegistryError(RToolError):
    """Unknown or conflicting agent/artifact registration."""


class ActionError(RToolError):
    """A queued human action cannot be applied (no open gate, bad index, ...)."""


class StageFailed(RToolError):
    """Raised by an agent to fail its stage with a human-readable reason.

    ``needs_human=True`` skips automatic retries and waits for ``rtool retry``.
    """

    def __init__(self, message: str, *, needs_human: bool = False) -> None:
        super().__init__(message)
        self.needs_human = needs_human


class BudgetExceeded(RToolError):
    """A token, call, GPU-hour or job budget was exhausted."""


class LeaseHeld(RToolError):
    """Another episode currently owns the run (or our lease was taken over)."""


class PathEscape(RToolError):
    """A workspace path resolved outside its jail."""


class ControlFlow(BaseException):
    """Base for signals agents use to hand control back to the engine.

    Deriving from ``BaseException`` means a broad ``except Exception`` inside
    agent code cannot swallow them by accident.
    """


class SuspendRequested(ControlFlow):
    """The agent wants to pause until the given Slurm jobs reach a terminal state."""

    def __init__(self, job_ids: list[str], reason: str = "") -> None:
        super().__init__(f"suspend until jobs {job_ids}")
        self.job_ids = job_ids
        self.reason = reason


class LLMRequired(ControlFlow):
    """The stage reached a step that needs the LLM, but this episode runs on a CPU node.

    The engine keeps the stage's state and hands it to a GPU episode, which resumes it.
    """


class GateRequested(ControlFlow):
    """The agent wants a human decision before continuing (an in-stage gate)."""

    def __init__(self, gate_id: str, summary: dict[str, Any] | None = None) -> None:
        super().__init__(f"gate {gate_id!r} requested")
        self.gate_id = gate_id
        self.summary = summary or {}
