"""Tool registry: lookup, the ``tools`` array sent to the model, and deferred loading.

Tools with ``defer_loading = True`` are not sent to the model until it loads them with the
built-in ``tool_search`` tool. That keeps the prompt small when you have many tools, which
matters a lot for open-weight models: tool-selection accuracy drops as the tool list grows.

The order of the ``tools`` array is the registration order and never changes between turns,
so vLLM's automatic prefix caching can reuse the rendered tool block.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from pydantic import Field

from .tool import Tool, ToolContext, ToolInput

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
TOOL_SEARCH = "tool_search"


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool[Any]] = ()) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        self._search = ToolSearchTool(self)
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool[Any]) -> None:
        if not _NAME_RE.match(tool.name):
            raise ValueError(f"Invalid tool name {tool.name!r}")
        if tool.name in self._tools or tool.name == TOOL_SEARCH:
            raise ValueError(f"Duplicate tool name {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any] | None:
        if name == TOOL_SEARCH and self.deferred():
            return self._search
        tool = self._tools.get(name)
        return tool if tool is not None and tool.is_enabled() else None

    def deferred(self) -> list[Tool[Any]]:
        return [t for t in self._tools.values() if t.defer_loading and t.is_enabled()]

    def is_visible(self, tool: Tool[Any], ctx: ToolContext) -> bool:
        return not tool.defer_loading or tool.name in ctx.loaded_tools

    def visible_names(self, ctx: ToolContext) -> list[str]:
        names = [t.name for t in self._tools.values() if t.is_enabled() and self.is_visible(t, ctx)]
        return names + ([TOOL_SEARCH] if self.deferred() else [])

    def openai_tools(self, ctx: ToolContext) -> list[dict[str, Any]]:
        """The ``tools`` parameter for an OpenAI-compatible chat completions request."""
        out = []
        for tool in self._tools.values():
            if tool.is_enabled() and self.is_visible(tool, ctx):
                out.append(to_openai_tool(tool))
        if self.deferred():
            out.append(to_openai_tool(self._search))
        return out


def to_openai_tool(tool: Tool[Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema(),
        },
    }


class ToolSearchInput(ToolInput):
    query: str = Field(
        description="Keywords describing the capability you need, or 'select:name1,name2' to load exact tools"
    )
    max_results: int = Field(5, ge=1, le=20)


class ToolSearchTool(Tool[ToolSearchInput]):
    name = TOOL_SEARCH
    Input = ToolSearchInput

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    @property
    def description(self) -> str:  # type: ignore[override]
        names = ", ".join(t.name for t in self._registry.deferred())
        return (
            "Load additional tools. These tools exist but their parameters are not loaded yet: "
            f"{names}. Call this with keywords or 'select:<name>' and the matching tools become "
            "callable from your next turn."
        )

    def is_read_only(self, inp: ToolSearchInput) -> bool:
        return True

    def is_concurrency_safe(self, inp: ToolSearchInput) -> bool:
        return True

    async def call(self, inp: ToolSearchInput, ctx: ToolContext) -> str:
        candidates = self._registry.deferred()
        query = inp.query.strip()
        if query.startswith("select:"):
            wanted = {n.strip() for n in query[len("select:") :].split(",") if n.strip()}
            matches = [t for t in candidates if t.name in wanted]
        else:
            words = [w for w in re.split(r"\W+", query.lower()) if w]
            scored = []
            for t in candidates:
                name, text = t.name.lower(), f"{t.search_hint} {t.description}".lower()
                score = sum(3 * (w in name) + (w in text) for w in words)
                if score:
                    scored.append((score, t))
            scored.sort(key=lambda st: -st[0])
            matches = [t for _, t in scored[: inp.max_results]]
        if not matches:
            return f"No tools matched {query!r}. Deferred tools: {', '.join(t.name for t in candidates)}"
        parts = []
        for t in matches:
            ctx.loaded_tools.add(t.name)
            parts.append(f"## {t.name}\n{t.description}\nParameters: {json.dumps(t.parameters_schema())}")
        return "Loaded these tools; you can call them now:\n\n" + "\n\n".join(parts)
