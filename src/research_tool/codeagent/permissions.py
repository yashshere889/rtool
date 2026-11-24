"""Permission engine: deny rules, the tool's own safety check, modes, allow rules, then a human.

Order of precedence for one call:

1. A matching **deny rule** -> deny.
2. The tool's ``check_permissions`` says deny -> deny.
3. ``plan`` mode and the call is not read-only -> deny.
4. ``bypass`` mode -> allow.
5. A pre-tool-use hook said deny -> deny; said allow -> allow (unless the tool asked).
6. The tool said allow -> allow.
7. Unless the tool asked: read-only call -> allow; every subject matched by an allow rule -> allow;
   ``accept_edits`` mode and an in-workspace file edit -> allow.
8. Otherwise ask the approver (a person). No approver -> deny.

Rules look like ``bash``, ``bash(git *)`` or ``edit_file(src/*)``. Patterns use fnmatch, where
``*`` also matches ``/``. A rule with a pattern allows a call only if *every* subject of the call
matches (so ``bash(git *)`` does not allow ``git status && rm -rf /``).
"""

from __future__ import annotations

import asyncio
import fnmatch
import inspect
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from .tool import Tool, ToolContext, ToolInput

Mode = Literal["default", "accept_edits", "plan", "bypass"]
MODES: tuple[str, ...] = ("default", "accept_edits", "plan", "bypass")

_RULE_RE = re.compile(r"\s*([A-Za-z0-9_\-*?]+)\s*(?:\((.*)\))?\s*", re.DOTALL)


@dataclass(frozen=True)
class Rule:
    tool: str
    pattern: str | None = None

    @classmethod
    def parse(cls, text: str) -> Rule:
        m = _RULE_RE.fullmatch(text)
        if not m:
            raise ValueError(f"Bad permission rule: {text!r} (expected 'tool' or 'tool(pattern)')")
        return cls(m.group(1), m.group(2))

    def applies_to(self, tool_name: str) -> bool:
        return fnmatch.fnmatchcase(tool_name, self.tool)

    def matches_subject(self, subject: str) -> bool:
        return self.pattern is None or fnmatch.fnmatchcase(subject, self.pattern)

    def __str__(self) -> str:
        return self.tool if self.pattern is None else f"{self.tool}({self.pattern})"


@dataclass
class ApprovalRequest:
    tool_name: str
    summary: str
    reason: str
    input: dict[str, Any]
    suggested_rule: str  # what "always allow" would add


@dataclass
class ApprovalResponse:
    approved: bool
    remember_rule: str | None = None  # add this allow rule for the rest of the session
    feedback: str = ""  # passed back to the model on denial


Approver = Callable[[ApprovalRequest], "Awaitable[ApprovalResponse] | ApprovalResponse"]


@dataclass
class Verdict:
    allowed: bool
    reason: str
    source: str  # which step decided: deny_rule, tool, mode, hook, read_only, allow_rule, user, ...


@dataclass
class PermissionEngine:
    mode: Mode = "default"
    allow: list[Rule] = field(default_factory=list)
    deny: list[Rule] = field(default_factory=list)
    approver: Approver | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"Unknown mode {self.mode!r}; choose one of {MODES}")
        self.allow = [r if isinstance(r, Rule) else Rule.parse(r) for r in self.allow]
        self.deny = [r if isinstance(r, Rule) else Rule.parse(r) for r in self.deny]
        self._prompt_lock = asyncio.Lock()  # one prompt at a time, even for parallel calls

    @classmethod
    def from_strings(
        cls,
        mode: Mode = "default",
        allow: Iterable[str] = (),
        deny: Iterable[str] = (),
        approver: Approver | None = None,
    ) -> PermissionEngine:
        return cls(mode, [Rule.parse(r) for r in allow], [Rule.parse(r) for r in deny], approver)

    async def decide(
        self,
        tool: Tool[Any],
        inp: ToolInput,
        ctx: ToolContext,
        hook_decision: Literal["allow", "deny", "ask"] | None = None,
    ) -> Verdict:
        subjects = tool.permission_subjects(inp)

        for rule in self.deny:
            if rule.applies_to(tool.name) and (
                rule.pattern is None or any(rule.matches_subject(s) for s in subjects or [])
            ):
                return Verdict(False, f"blocked by deny rule {rule}", "deny_rule")

        decision = await tool.check_permissions(inp, ctx)
        if decision.behavior == "deny":
            return Verdict(False, decision.reason or "refused by the tool", "tool")

        read_only = _safe_flag(tool.is_read_only, inp)
        if self.mode == "plan" and not read_only:
            return Verdict(False, "plan mode allows read-only tools only", "mode")
        if self.mode == "bypass":
            return Verdict(True, "bypass mode", "mode")

        if hook_decision == "deny":
            return Verdict(False, "denied by a pre_tool_use hook", "hook")
        if hook_decision == "allow" and decision.behavior != "ask":
            return Verdict(True, "allowed by a pre_tool_use hook", "hook")
        if decision.behavior == "allow":
            return Verdict(True, decision.reason or "allowed by the tool", "tool")

        if decision.behavior != "ask" and hook_decision != "ask":
            if read_only:
                return Verdict(True, "read-only", "read_only")
            if self._allowed_by_rules(tool.name, subjects):
                return Verdict(True, "matched allow rules", "allow_rule")
            if self.mode == "accept_edits" and _safe_flag(tool.is_file_edit, inp):
                return Verdict(True, "accept_edits mode", "mode")

        reason = decision.reason or f"{tool.name} needs approval"
        return await self._ask(tool, inp, subjects, reason)

    def _allowed_by_rules(self, tool_name: str, subjects: list[str] | None) -> bool:
        rules = [r for r in self.allow if r.applies_to(tool_name)]
        if any(r.pattern is None for r in rules):
            return True
        if not subjects or not rules:
            return False
        return all(any(r.matches_subject(s) for r in rules) for s in subjects)

    async def _ask(self, tool: Tool[Any], inp: ToolInput, subjects: list[str] | None, reason: str) -> Verdict:
        if self.approver is None:
            return Verdict(False, f"{reason}, and no approver is configured", "no_approver")
        suggested = tool.name if not subjects or len(subjects) != 1 else f"{tool.name}({subjects[0]})"
        request = ApprovalRequest(
            tool_name=tool.name,
            summary=tool.describe_call(inp),
            reason=reason,
            input=inp.model_dump(),
            suggested_rule=suggested,
        )
        async with self._prompt_lock:
            response = self.approver(request)
            if inspect.isawaitable(response):
                response = await response
        if response.remember_rule:
            self.allow.append(Rule.parse(response.remember_rule))
        if response.approved:
            return Verdict(True, "approved by the user", "user")
        extra = f": {response.feedback}" if response.feedback else ""
        return Verdict(False, f"the user declined{extra}", "user")


def _safe_flag(fn: Callable[[Any], bool], inp: ToolInput) -> bool:
    try:
        return bool(fn(inp))
    except Exception:
        return False  # a buggy flag must never make a call look safer
