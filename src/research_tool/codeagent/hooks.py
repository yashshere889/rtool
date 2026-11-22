"""Hooks: your code runs around every tool call.

``pre_tool_use`` hooks may block a call, rewrite its input (it is re-validated), or vote on
permission. ``post_tool_use`` hooks see every result (``result.is_error`` tells you if it failed)
and may replace it. Hooks can be sync or async; the matcher is an fnmatch pattern on tool names.

    hooks = Hooks()

    @hooks.pre_tool_use("bash")
    def no_force_push(tool, inp, ctx):
        if "push --force" in inp["command"]:
            return PreToolUseOutcome(block="force pushes are not allowed")
"""

from __future__ import annotations

import fnmatch
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .tool import Tool, ToolContext, ToolResult

log = logging.getLogger(__name__)


@dataclass
class PreToolUseOutcome:
    block: str | None = None  # reason; the call is not run and the model sees this reason
    updated_input: dict[str, Any] | None = None
    permission: Literal["allow", "deny", "ask"] | None = None


PreHook = Callable[[Tool[Any], dict[str, Any], ToolContext], Any]
PostHook = Callable[[Tool[Any], dict[str, Any], ToolResult, ToolContext], Any]

_STRICTNESS = {None: 0, "allow": 1, "ask": 2, "deny": 3}


class Hooks:
    def __init__(self) -> None:
        self._pre: list[tuple[str, PreHook]] = []
        self._post: list[tuple[str, PostHook]] = []

    def pre_tool_use(self, matcher: str = "*") -> Callable[[PreHook], PreHook]:
        def register(fn: PreHook) -> PreHook:
            self._pre.append((matcher, fn))
            return fn

        return register

    def post_tool_use(self, matcher: str = "*") -> Callable[[PostHook], PostHook]:
        def register(fn: PostHook) -> PostHook:
            self._post.append((matcher, fn))
            return fn

        return register

    async def run_pre(self, tool: Tool[Any], inp: dict[str, Any], ctx: ToolContext) -> PreToolUseOutcome:
        combined = PreToolUseOutcome()
        current = inp
        for matcher, fn in self._pre:
            if not fnmatch.fnmatchcase(tool.name, matcher):
                continue
            try:
                out = fn(tool, current, ctx)
                if inspect.isawaitable(out):
                    out = await out
            except Exception as exc:  # fail closed: a broken guard must not let the call through
                log.exception("pre_tool_use hook %s failed", getattr(fn, "__name__", fn))
                return PreToolUseOutcome(block=f"a pre_tool_use hook failed ({type(exc).__name__}: {exc})")
            if out is None:
                continue
            if out.block:
                return PreToolUseOutcome(block=out.block)
            if out.updated_input is not None:
                current = out.updated_input
                combined.updated_input = current
            if _STRICTNESS[out.permission] > _STRICTNESS[combined.permission]:
                combined.permission = out.permission
        return combined

    async def run_post(self, tool: Tool[Any], inp: dict[str, Any], result: ToolResult, ctx: ToolContext) -> ToolResult:
        for matcher, fn in self._post:
            if not fnmatch.fnmatchcase(tool.name, matcher):
                continue
            try:
                out = fn(tool, inp, result, ctx)
                if inspect.isawaitable(out):
                    out = await out
            except Exception:
                log.exception("post_tool_use hook %s failed; keeping the result", getattr(fn, "__name__", fn))
                continue
            if isinstance(out, ToolResult):
                result = out
        return result
