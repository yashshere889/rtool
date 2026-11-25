"""Core types every tool and the executor share.

A tool is a small class: a pydantic input model, a description for the model,
a few *per-call* safety flags, and an async ``call``. The defaults are
deliberately conservative ("fail closed"): unless a tool says otherwise it is
treated as writing, not safe to run in parallel, and in need of approval.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict


class ToolInput(BaseModel):
    """Base class for tool inputs. Unknown fields are rejected so typos surface as errors."""

    model_config = ConfigDict(extra="forbid")


class ToolError(Exception):
    """Raise inside a tool to send an error message back to the model; the turn continues."""


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)  # for your UI/logs, never sent to the model

    @classmethod
    def error(cls, message: str) -> ToolResult:
        return cls(content=message, is_error=True)


Behavior = Literal["allow", "deny", "ask", "passthrough"]


@dataclass
class PermissionDecision:
    """What a tool's own ``check_permissions`` thinks about one call.

    ``passthrough`` means "no opinion, let rules and the permission mode decide".
    ``ask`` is a safety signal: allow rules cannot override it, only a human (or bypass mode).
    """

    behavior: Behavior
    reason: str = ""

    @classmethod
    def allow(cls, reason: str = "") -> PermissionDecision:
        return cls("allow", reason)

    @classmethod
    def deny(cls, reason: str) -> PermissionDecision:
        return cls("deny", reason)

    @classmethod
    def ask(cls, reason: str) -> PermissionDecision:
        return cls("ask", reason)

    @classmethod
    def passthrough(cls) -> PermissionDecision:
        return cls("passthrough")


class CancelToken:
    """A cancellation flag that can be set from any thread and awaited from any event loop.

    (``asyncio.Event`` binds to the first loop that waits on it, which breaks a context that
    is reused across ``asyncio.run`` calls or cancelled from a UI thread.)
    """

    def __init__(self) -> None:
        self._flag = threading.Event()
        self._lock = threading.Lock()
        self._waiters: set[tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]] = set()

    def is_set(self) -> bool:
        return self._flag.is_set()

    def set(self) -> None:
        with self._lock:
            self._flag.set()
            waiters, self._waiters = self._waiters, set()
        for loop, fut in waiters:
            if not loop.is_closed():
                loop.call_soon_threadsafe(_resolve, fut)

    def clear(self) -> None:
        self._flag.clear()

    async def wait(self) -> None:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        entry = (loop, fut)
        with self._lock:
            if self._flag.is_set():
                return
            self._waiters.add(entry)
        try:
            await fut
        finally:
            with self._lock:
                self._waiters.discard(entry)


def _resolve(fut: asyncio.Future[None]) -> None:
    if not fut.done():
        fut.set_result(None)


@dataclass
class ToolContext:
    """Per-session state handed to every tool call."""

    workspace: Path
    cwd: Path | None = None
    scratch_dir: Path | None = None  # where oversized tool outputs are saved
    cancel: CancelToken = field(default_factory=CancelToken)
    read_files: dict[Path, int] = field(default_factory=dict)  # path -> mtime_ns when last read
    loaded_tools: set[str] = field(default_factory=set)  # deferred tools loaded via tool_search
    state: dict[str, Any] = field(default_factory=dict)  # free-form, for your own tools
    env: dict[str, str] | None = None  # environment of the processes tools start (None: this process's own)

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).resolve()
        self.cwd = Path(self.cwd).resolve() if self.cwd else self.workspace
        self.scratch_dir = (
            Path(self.scratch_dir).resolve() if self.scratch_dir else self.workspace / ".agent" / "tool-results"
        )

    def resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.cwd / p
        return p.resolve()

    def in_workspace(self, p: Path) -> bool:
        return p.is_relative_to(self.workspace)

    def freely_readable(self, p: Path) -> bool:
        return self.in_workspace(p) or p.is_relative_to(self.scratch_dir)

    def display(self, p: Path) -> str:
        return str(p.relative_to(self.workspace)) if self.in_workspace(p) else str(p)


InputT = TypeVar("InputT", bound=ToolInput)


class Tool(Generic[InputT]):
    """Subclass this to add a tool. Only ``name``, ``description``, ``Input`` and ``call`` are required."""

    name: ClassVar[str]
    description: str  # what the model reads; a property is fine if it must be dynamic
    Input: ClassVar[type[ToolInput]]

    search_hint: ClassVar[str] = ""  # extra keywords for tool_search (deferred tools)
    defer_loading: ClassVar[bool] = False  # hide the schema until tool_search loads it
    max_result_chars: ClassVar[int] = 30_000  # larger outputs are saved to a file and previewed
    timeout_s: ClassVar[float | None] = None  # executor-enforced timeout; None = tool manages its own

    # ---- per-call flags (they receive the validated input) -------------------------------
    def is_enabled(self) -> bool:
        return True

    def is_read_only(self, inp: InputT) -> bool:
        return False

    def is_concurrency_safe(self, inp: InputT) -> bool:
        """True if this call may run in parallel with other concurrency-safe calls."""
        return False

    def is_file_edit(self, inp: InputT) -> bool:
        """True for in-workspace file edits; auto-allowed in ``accept_edits`` mode."""
        return False

    def permission_subjects(self, inp: InputT) -> list[str] | None:
        """Strings that permission rules like ``bash(git *)`` are matched against."""
        return None

    # ---- checks ---------------------------------------------------------------------------
    async def validate(self, inp: InputT, ctx: ToolContext) -> str | None:
        """Return an error message to reject the call before any permission prompt."""
        return None

    async def check_permissions(self, inp: InputT, ctx: ToolContext) -> PermissionDecision:
        return PermissionDecision.passthrough()

    # ---- execution ------------------------------------------------------------------------
    async def call(self, inp: InputT, ctx: ToolContext) -> ToolResult | str:
        raise NotImplementedError

    def describe_call(self, inp: InputT) -> str:
        """One line for approval prompts and logs."""
        return f"{self.name} {inp.model_dump_json(exclude_none=True)[:200]}"

    def parameters_schema(self) -> dict[str, Any]:
        return clean_schema(self.Input.model_json_schema())


def clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Shrink pydantic's JSON Schema: drop titles and collapse ``Optional[X]`` to ``X``.

    Open-weight models' chat templates handle flat schemas far better than ``anyOf`` unions.
    """

    def walk(node: Any, is_props: bool = False) -> Any:
        if isinstance(node, list):
            return [walk(v) for v in node]
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "title" and not is_props:
                continue
            out[key] = walk(value, is_props=(key == "properties" and not is_props))
        if not is_props:
            any_of = out.get("anyOf")
            if isinstance(any_of, list) and len(any_of) == 2 and {"type": "null"} in any_of:
                other = next(x for x in any_of if x != {"type": "null"})
                rest = {k: v for k, v in out.items() if k != "anyOf"}
                out = {**other, **rest}
        return out

    return walk(schema)
