"""Base classes for pipeline stages.

An agent declares its contract as class attributes and implements ``run``::

    @register_agent("summarizer")
    class Summarizer(Agent):
        consumes = ("corpus",)
        produces = ("summary",)

        class Config(BaseModel):
            max_words: int = 200
        config_model = Config

        def run(self, ctx: AgentContext) -> None:
            corpus = ctx.inputs.get("corpus")
            ...
            ctx.outputs.put("summary", Summary(text=...))

Rules that keep the system easy to change:

* talk to other stages only through artifacts (``ctx.inputs`` / ``ctx.outputs``);
* keep anything you need after a suspend/resume in ``ctx.state`` (JSON-serialisable);
* never refer to other stages by id — use ``ctx.request_change(artifact=...)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from .context import AgentContext


class Agent(ABC):
    name: ClassVar[str] = ""
    version: ClassVar[str] = "0.1.0"
    description: ClassVar[str] = ""
    consumes: ClassVar[tuple[str, ...]] = ()
    optional_inputs: ClassVar[tuple[str, ...]] = ()
    produces: ClassVar[tuple[str, ...]] = ()
    needs_llm: ClassVar[bool] = True
    config_model: ClassVar[type[BaseModel] | None] = None
    # workspace-relative files a researcher may rewrite while the stage waits at its own checkpoint
    # or after it failed (the agent must cope with the edit when it resumes)
    editable_files: ClassVar[tuple[str, ...]] = ()

    def __init__(self, config: BaseModel | dict[str, Any] | None = None) -> None:
        if config is None:
            config = self.validate_config({})
        self.config: Any = config

    @classmethod
    def validate_config(cls, raw: dict[str, Any]) -> BaseModel | dict[str, Any]:
        """Validate the ``config:`` block of a pipeline stage (called when the pipeline loads)."""
        if cls.config_model is None:
            if raw:
                raise ValueError(f"agent {cls.name!r} takes no config, got keys {sorted(raw)}")
            return {}
        try:
            return cls.config_model.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"invalid config for agent {cls.name!r}: {exc}") from None

    @abstractmethod
    def run(self, ctx: AgentContext) -> None:
        """Do the stage's work. Return normally on success; raise StageFailed to fail."""


class Task(Agent):
    """A deterministic stage that never calls the LLM (downloads, builds, checks)."""

    needs_llm = False
