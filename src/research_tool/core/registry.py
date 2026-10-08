"""Agent registry: how new agents plug in without touching the engine.

Built-in agents register themselves with ``@register_agent("name")`` when
their module is imported. Third-party packages can add agents by exposing an
entry point in the ``research_tool.agents`` group that imports their module.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, TypeVar

from .errors import RegistryError

if TYPE_CHECKING:
    from .agent import Agent

A = TypeVar("A", bound="type[Agent]")

# Modules imported on first lookup so their agents are always available.
BUILTIN_AGENT_MODULES: list[str] = [
    "research_tool.testing.toy_agents",
    "research_tool.agents.literature",
    "research_tool.agents.hypothesis",
    "research_tool.agents.planner",
    "research_tool.agents.tasks",
    "research_tool.agents.coder",
    "research_tool.agents.writer",
    "research_tool.agents.reviewer",
]

ENTRY_POINT_GROUP = "research_tool.agents"


def _qualname(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


class AgentRegistry:
    def __init__(self, builtin_modules: list[str] | None = None) -> None:
        self._agents: dict[str, type[Agent]] = {}
        self._builtin_modules = list(BUILTIN_AGENT_MODULES if builtin_modules is None else builtin_modules)
        self._plugins_loaded = False

    def register(self, name: str | None = None) -> Callable[[A], A]:
        def decorate(cls: A) -> A:
            key = name or getattr(cls, "name", "")
            if not key:
                raise RegistryError(f"{_qualname(cls)} needs a name: @register_agent('my_agent')")
            existing = self._agents.get(key)
            if existing is not None and _qualname(existing) != _qualname(cls):
                raise RegistryError(f"agent {key!r} is already registered by {_qualname(existing)}")
            cls.name = key
            self._agents[key] = cls
            return cls

        return decorate

    def load_plugins(self) -> None:
        if self._plugins_loaded:
            return
        self._plugins_loaded = True
        for module in self._builtin_modules:
            import_module(module)
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            ep.load()

    def get(self, name: str) -> type[Agent]:
        if name not in self._agents:
            self.load_plugins()
        try:
            return self._agents[name]
        except KeyError:
            raise RegistryError(f"unknown agent {name!r}; registered agents: {self.names()}") from None

    def names(self) -> list[str]:
        self.load_plugins()
        return sorted(self._agents)

    def items(self) -> list[tuple[str, type[Agent]]]:
        self.load_plugins()
        return sorted(self._agents.items())


registry = AgentRegistry()
register_agent = registry.register
