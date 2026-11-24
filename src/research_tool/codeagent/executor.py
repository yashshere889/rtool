"""Tool executor: schedules a turn's tool calls and runs each through the full pipeline.

Scheduling: consecutive calls that are concurrency-safe *for their input* (reads, searches)
form one batch that runs in parallel; anything else runs alone, in order. So
``[read, grep, edit, read]`` runs as ``{read, grep}`` in parallel, then ``edit``, then ``read``.
Results always come back in the order the model asked for them.

Pipeline for one call (every step can end it with an error result the model can react to):

    lookup -> parse JSON -> validate schema -> tool.validate -> pre_tool_use hooks
    -> permission engine -> call (timeout + cancellation) -> post_tool_use hooks
    -> output budget (oversized output saved to a file, preview returned)

Invariant: every call gets exactly one result. Nothing in here raises except cancellation of
the whole turn, because a missing tool result breaks the conversation for the model.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .hooks import Hooks
from .permissions import PermissionEngine
from .registry import ToolRegistry
from .tool import Tool, ToolContext, ToolError, ToolInput, ToolResult

log = logging.getLogger(__name__)

EventSink = Callable[[dict[str, Any]], None]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str | dict[str, Any]  # raw JSON string as the model produced it, or a parsed dict


@dataclass
class ToolOutcome:
    call: ToolCall
    result: ToolResult
    duration_s: float = 0.0


@dataclass
class _Prepared:
    call: ToolCall
    tool: Tool[Any] | None = None
    inp: ToolInput | None = None
    error: str | None = None

    def concurrency_safe(self) -> bool:
        if self.error is not None or self.tool is None or self.inp is None:
            return True  # resolves instantly to an error, harmless to batch
        try:
            return bool(self.tool.is_concurrency_safe(self.inp))
        except Exception:
            return False


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        permissions: PermissionEngine,
        hooks: Hooks | None = None,
        *,
        max_concurrency: int = 8,
        on_event: EventSink | None = None,
    ) -> None:
        self.registry = registry
        self.permissions = permissions
        self.hooks = hooks or Hooks()
        self.max_concurrency = max_concurrency
        self.on_event = on_event

    # ---- scheduling ---------------------------------------------------------------------
    async def run(self, calls: list[ToolCall], ctx: ToolContext) -> list[ToolOutcome]:
        prepared = [self._prepare(c, ctx) for c in calls]
        outcomes: list[ToolOutcome | None] = [None] * len(prepared)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_at(i: int) -> None:
            async with semaphore:
                outcomes[i] = await self._execute(prepared[i], ctx)

        for parallel, indexes in partition([p.concurrency_safe() for p in prepared]):
            if parallel:
                await asyncio.gather(*(run_at(i) for i in indexes))
            else:
                for i in indexes:
                    outcomes[i] = await self._execute(prepared[i], ctx)
        return [o for o in outcomes if o is not None]

    # ---- one call -----------------------------------------------------------------------
    def _prepare(self, call: ToolCall, ctx: ToolContext) -> _Prepared:
        tool = self.registry.get(call.name)
        if tool is None:
            names = self.registry.visible_names(ctx)
            close = difflib.get_close_matches(call.name, names, n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            return _Prepared(call, error=f"Unknown tool '{call.name}'.{hint} Available tools: {', '.join(names)}")
        if not self.registry.is_visible(tool, ctx):
            return _Prepared(
                call,
                tool,
                error=f"Tool '{tool.name}' is not loaded. Call tool_search with query 'select:{tool.name}' first.",
            )
        try:
            args = parse_arguments(call.arguments)
        except ValueError as exc:
            return _Prepared(
                call,
                tool,
                error=f"Could not parse the arguments for {tool.name} as a JSON object ({exc}). "
                "Send one JSON object that matches the tool's parameters.",
            )
        try:
            return _Prepared(call, tool, tool.Input.model_validate(args))
        except ValidationError as exc:
            return _Prepared(call, tool, error=format_validation_error(tool.name, exc))

    async def _execute(self, p: _Prepared, ctx: ToolContext) -> ToolOutcome:
        started = time.monotonic()
        summary = p.tool.describe_call(p.inp) if p.tool and p.inp else p.call.name
        self._emit({"type": "tool_start", "id": p.call.id, "name": p.call.name, "summary": summary})
        try:
            result = await self._pipeline(p, ctx)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # last line of defence; the pipeline should not raise
            log.exception("internal error running %s", p.call.name)
            result = ToolResult.error(f"Internal error while running {p.call.name}: {type(exc).__name__}: {exc}")
        duration = time.monotonic() - started
        self._emit(
            {
                "type": "tool_end",
                "id": p.call.id,
                "name": p.call.name,
                "is_error": result.is_error,
                "duration_s": round(duration, 3),
            }
        )
        return ToolOutcome(p.call, result, duration)

    async def _pipeline(self, p: _Prepared, ctx: ToolContext) -> ToolResult:
        if p.error is not None:
            return ToolResult.error(p.error)
        assert p.tool is not None and p.inp is not None
        tool, inp = p.tool, p.inp

        if ctx.cancel.is_set():
            return ToolResult.error("Cancelled by the user before it started.")

        if message := await tool.validate(inp, ctx):
            return ToolResult.error(message)

        pre = await self.hooks.run_pre(tool, inp.model_dump(), ctx)
        if pre.block:
            return ToolResult.error(f"Blocked by a hook: {pre.block}")
        if pre.updated_input is not None:
            try:
                inp = tool.Input.model_validate(pre.updated_input)
            except ValidationError as exc:
                return ToolResult.error(
                    "A hook rewrote the input into something invalid.\n" + format_validation_error(tool.name, exc)
                )
            if message := await tool.validate(inp, ctx):
                return ToolResult.error(message)

        verdict = await self.permissions.decide(tool, inp, ctx, hook_decision=pre.permission)
        if not verdict.allowed:
            return ToolResult.error(
                f"Permission denied: {verdict.reason}. Do not retry this call or try to work around it; "
                "continue with something else or tell the user what you need."
            )

        result = await self._invoke(tool, inp, ctx)
        result = await self.hooks.run_post(tool, inp.model_dump(), result, ctx)
        return self._apply_budget(tool, p.call, result, ctx)

    async def _invoke(self, tool: Tool[Any], inp: ToolInput, ctx: ToolContext) -> ToolResult:
        task = asyncio.create_task(tool.call(inp, ctx))
        cancel_wait = asyncio.create_task(ctx.cancel.wait())
        try:
            done, _ = await asyncio.wait(
                {task, cancel_wait}, timeout=tool.timeout_s, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            task.cancel()
            raise
        finally:
            cancel_wait.cancel()

        if task in done:
            try:
                out = task.result()
            except ToolError as exc:
                return ToolResult.error(str(exc))
            except Exception as exc:
                log.exception("tool %s raised", tool.name)
                return ToolResult.error(f"{tool.name} failed: {type(exc).__name__}: {exc}")
            return out if isinstance(out, ToolResult) else ToolResult(str(out))

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        if ctx.cancel.is_set():
            return ToolResult.error("Interrupted by the user.")
        if tool.timeout_s is not None:
            return ToolResult.error(f"{tool.name} timed out after {tool.timeout_s}s.")
        return ToolResult.error(f"{tool.name} stopped unexpectedly (its cancellation watcher failed).")

    def _apply_budget(self, tool: Tool[Any], call: ToolCall, result: ToolResult, ctx: ToolContext) -> ToolResult:
        limit = tool.max_result_chars
        text = result.content
        if len(text) <= limit:
            return result
        assert ctx.scratch_dir is not None
        ctx.scratch_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", call.id)[:40]
        path = ctx.scratch_dir / f"{tool.name}-{safe_id}-{uuid.uuid4().hex[:8]}.txt"
        path.write_text(text, encoding="utf-8")
        head, tail = text[: int(limit * 0.7)], text[-int(limit * 0.2) :]
        omitted = len(text) - len(head) - len(tail)
        preview = (
            f"{head}\n\n... [{omitted} characters omitted] ...\n\n{tail}\n\n"
            f"[The output was {len(text)} characters. The full text is saved at {path}; "
            "use read_file with offset/limit, or grep, to look at the rest.]"
        )
        return ToolResult(preview, result.is_error, {**result.metadata, "full_output_path": str(path)})

    def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                log.exception("on_event callback failed")


def partition(safe_flags: list[bool]) -> list[tuple[bool, list[int]]]:
    """Group consecutive concurrency-safe calls; every unsafe call gets its own group."""
    groups: list[tuple[bool, list[int]]] = []
    for i, safe in enumerate(safe_flags):
        if safe and groups and groups[-1][0]:
            groups[-1][1].append(i)
        else:
            groups.append((safe, [i]))
    return groups


def parse_arguments(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None or not raw.strip():
        return {}
    value = json.loads(raw)
    if isinstance(value, str):  # some models double-encode their arguments
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object, got {type(value).__name__}")
    return value


def format_validation_error(tool_name: str, exc: ValidationError) -> str:
    lines = []
    for err in exc.errors(include_url=False):
        loc = ".".join(str(x) for x in err["loc"]) or "(input)"
        lines.append(f"- {loc}: {err['msg']}")
    return f"Invalid input for {tool_name}:\n" + "\n".join(lines)
