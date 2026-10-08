"""Usage accounting and budget enforcement (per stage and per run)."""

from __future__ import annotations

import threading

from pydantic import BaseModel

from .config import BudgetLimits
from .errors import BudgetExceeded


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    gpu_hours: float = 0.0
    jobs: int = 0
    wall_seconds: float = 0.0

    def plus(self, **amounts: float) -> Usage:
        data = self.model_dump()
        for key, amount in amounts.items():
            if key not in data:
                raise KeyError(f"unknown usage field {key!r}")
            data[key] = data[key] + amount
        return Usage.model_validate(data)


# usage field -> limit field
_LIMITS = {
    "prompt_tokens": "max_prompt_tokens",
    "completion_tokens": "max_completion_tokens",
    "llm_calls": "max_llm_calls",
    "gpu_hours": "max_gpu_hours",
    "jobs": "max_jobs",
}


def _check(usage: Usage, limits: BudgetLimits, label: str) -> None:
    for field, limit_field in _LIMITS.items():
        limit = getattr(limits, limit_field)
        value = getattr(usage, field)
        if limit is not None and value > limit:
            raise BudgetExceeded(f"{label} budget exceeded: {field} {value:g} > {limit:g}")


class BudgetTracker:
    """Charges usage to a stage and to the whole run; raises when either limit is crossed."""

    def __init__(
        self,
        stage_limits: BudgetLimits,
        run_limits: BudgetLimits,
        stage_usage: Usage | None = None,
        run_usage: Usage | None = None,
        label: str = "stage",
    ) -> None:
        self.stage_limits = stage_limits
        self.run_limits = run_limits
        self.stage_usage = stage_usage or Usage()
        self.run_usage = run_usage or Usage()
        self.label = label
        self._lock = threading.Lock()  # bulk LLM calls charge from worker threads

    def charge(self, **amounts: float) -> None:
        with self._lock:
            self.stage_usage = self.stage_usage.plus(**amounts)
            self.run_usage = self.run_usage.plus(**amounts)
            _check(self.stage_usage, self.stage_limits, self.label)
            _check(self.run_usage, self.run_limits, "run")

    def would_exceed(self, **amounts: float) -> str | None:
        """Return a message if charging ``amounts`` would cross a limit (no side effects)."""
        with self._lock:
            try:
                _check(self.stage_usage.plus(**amounts), self.stage_limits, self.label)
                _check(self.run_usage.plus(**amounts), self.run_limits, "run")
            except BudgetExceeded as exc:
                return str(exc)
        return None
