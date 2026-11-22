"""Hand any task to a tool-calling loop: the vendored codeagent harness (read/edit/write/glob/grep/bash).

    result = run_tool_agent(ctx, "Summarise the CSV files in ./data into notes.md", workspace=folder)
    if result.stopped: ...  # the model did not finish on its own

Every model turn goes through ``ctx.llm`` (profile, stage budget, ``llm_call`` events, retries). File tools
work inside ``workspace``; read-only tools may also read under ``read_roots``. Agents can add their own
tools (any ``research_tool.codeagent.Tool``), e.g. one that runs code in the experiment container.

No person is present in an episode job, so a call the rules do not allow is refused with a reason the
model can act on, and the loop stops after the same refused call five times (Qwen repeated one up to
147 times in the 2026-11-19 pilots). The transcript goes to ``<stage workspace>/logs/<name>.jsonl``.

The LLM server must parse tool calls (``--enable-auto-tool-choice --tool-call-parser ...`` in
``llm.server.extra_args``).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..codeagent import (
    DEFAULT_SYSTEM_PROMPT,
    ApprovalRequest,
    ApprovalResponse,
    Hooks,
    PermissionEngine,
    RToolModel,
    Rule,
    Tool,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    default_tools,
)
from ..codeagent import Agent as ToolLoop
from ..codeagent.tools.bash import _subject, split_command  # the same split the permission engine uses
from ..llm.vllm_backend import LLMRequestError
from .base import Agent, AgentContext, StageFailed, artifact_type, register_agent

# Shell commands any task may run besides the bash tool's read-only list. Qwen prefixes most commands
# with `cd <workspace> &&` and codeagent checks each part, so cd must pass (2026-11-19: 21 of 23 refused).
ALLOW = ("bash(cd *)", "bash(mkdir *)")
# rtool owns downloads, installs and Slurm jobs; these stay refused whatever a caller allows.
DENY = (
    "bash(curl *)",
    "bash(wget *)",
    "bash(pip *)",
    "bash(* -m pip *)",
    "bash(rm -rf *)",
    "bash(ssh *)",
    "bash(scp *)",
    "bash(git push*)",
    "bash(sbatch *)",
    "bash(scancel *)",
)
READ_TOOLS = ("read_file", "glob", "grep")
MAX_SAME_DENIAL = 5


@dataclass
class ToolAgentResult:
    text: str  # the model's final reply
    stopped: str | None  # None: the model finished; else max_turns, max_tokens, repeated_denial or context_full
    turns: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)
    tool_errors: dict[str, int] = field(default_factory=dict)
    denials: list[str] = field(default_factory=list)
    transcript: Path | None = None


def run_tool_agent(
    ctx: AgentContext,
    task: str,
    *,
    workspace: Path,
    tools: Sequence[Tool[Any]] = (),
    allow: Iterable[str] = (),
    deny: Iterable[str] = (),
    read_roots: Iterable[Path] = (),
    system: str | None = None,
    profile: str = "code",
    max_turns: int = 40,
    name: str = "tool-agent",
) -> ToolAgentResult:
    """Run ``task`` to completion with the default tools plus ``tools``; blocks until the loop ends.

    shortcut: no resume; a preempted episode starts the loop again from the files already in ``workspace``.
    """
    return asyncio.run(
        _run(ctx, task, Path(workspace), list(tools), list(allow), list(deny), [Path(p) for p in read_roots],
             system, profile, max_turns, name)
    )  # fmt: skip


async def _run(
    ctx: AgentContext,
    task: str,
    workspace: Path,
    tools: list[Tool[Any]],
    allow: list[str],
    deny: list[str],
    read_roots: list[Path],
    system: str | None,
    profile: str,
    max_turns: int,
    name: str,
) -> ToolAgentResult:
    tctx = ToolContext(workspace=workspace, scratch_dir=ctx.workspace.mkdir("scratch") / name)
    allow, deny = [*ALLOW, *allow], [*DENY, *deny]
    allow_rules = [Rule.parse(r) for r in allow]
    roots = [p.resolve() for p in read_roots]
    transcript = ctx.workspace.mkdir("logs") / f"{name}.jsonl"
    calls, errors, denials = Counter(), Counter(), []
    stopped: str | None = None
    t0 = time.monotonic()

    with transcript.open("w", encoding="utf-8") as log:

        def on_event(event: dict[str, Any]) -> None:
            log.write(json.dumps({"t": round(time.monotonic() - t0, 1), **event}, default=str) + "\n")
            log.flush()
            if event["type"] == "tool_start":
                calls[event["name"]] += 1
            elif event["type"] == "tool_end" and event.get("is_error"):
                errors[event["name"]] += 1

        def approver(req: ApprovalRequest) -> ApprovalResponse:
            nonlocal stopped
            if req.tool_name in READ_TOOLS:
                target = tctx.resolve(str(req.input.get("path") or "."))
                if any(target.is_relative_to(root) for root in roots):
                    return ApprovalResponse(True)
            denials.append(req.summary)
            if denials.count(req.summary) >= MAX_SAME_DENIAL:
                stopped = "repeated_denial"
                tctx.cancel.set()
            why = ""
            if req.tool_name == "bash":
                parts = split_command(str(req.input.get("command", ""))) or []
                bad = [s for s in map(_subject, parts) if not any(r.matches_subject(s) for r in allow_rules)]
                why = f" Not allowed: {bad}." if bad else " The command could not be analysed (no $(...) or backticks)."
            readable = f" Read-only tools may also read under {[str(r) for r in roots]}." if roots else ""
            return ApprovalResponse(
                False,
                feedback=(
                    f"no person is available in this batch job.{why} Each part of a command joined by &&, ; or | "
                    f"must match one of {allow} or be a read-only command; file tools work inside the "
                    f"workspace {tctx.workspace}.{readable} Do not repeat a refused call; change it."
                ),
            )

        registry = ToolRegistry([*default_tools(), *tools])
        permissions = PermissionEngine.from_strings("accept_edits", allow, deny, approver)
        executor = ToolExecutor(registry, permissions, Hooks(), on_event=on_event)
        model = RToolModel(ctx.llm, profile=profile)
        agent = ToolLoop(
            model, registry, executor, tctx, system_prompt=system or DEFAULT_SYSTEM_PROMPT,
            max_turns=max_turns, on_event=on_event,
        )  # fmt: skip
        try:
            text = await agent.send(task)
        except LLMRequestError as exc:
            if "context length" not in str(exc):
                raise
            stopped, text = "context_full", f"[stopped: the conversation outgrew the model's context ({exc})]"
        if stopped is None and text.startswith("[stopped: reached max_turns"):
            stopped = "max_turns"
        elif stopped is None and text.endswith("[stopped: the model hit max_tokens]"):
            stopped = "max_tokens"

    turns = sum(1 for m in agent.messages if m.get("role") == "assistant")
    result = ToolAgentResult(text, stopped, turns, dict(calls), dict(errors), denials, transcript)
    ctx.emit(
        "tool_agent",
        name=name,
        turns=turns,
        stopped=stopped,
        tool_calls=result.tool_calls,
        tool_errors=result.tool_errors,
        denials=len(denials),
        seconds=round(time.monotonic() - t0, 1),
    )
    return result


# ------------------------------------------------------------------ a pipeline stage for any task
@artifact_type("tool_task_result")
class ToolTaskResult(BaseModel):
    task: str
    reply: str
    files: list[str] = Field(default_factory=list)  # run-relative paths the agent left in its folder
    turns: int = 0
    tool_calls: dict[str, int] = Field(default_factory=dict)


class ToolTaskConfig(BaseModel):
    task: str = Field(min_length=1)  # "{topic}" is replaced by the research brief's topic
    allow: list[str] = Field(default_factory=list)  # extra permission rules, e.g. "bash(python *)"
    max_turns: int = Field(default=40, ge=1, le=200)
    profile: str = "code"


@register_agent("tool_task")
class ToolTask(Agent):
    description = "Hands a free-form task to the tool-calling loop (files and shell) and keeps what it produced."
    consumes = ("research_brief",)
    produces = ("tool_task_result",)
    config_model = ToolTaskConfig

    def run(self, ctx: AgentContext) -> None:
        cfg: ToolTaskConfig = self.config
        task = cfg.task.replace("{topic}", ctx.inputs.get("research_brief").topic)
        feedback = [f.text for f in ctx.feedback]
        if feedback:
            task += "\n\nA reviewer asked for these changes to an earlier attempt:\n" + "\n".join(
                f"- {f}" for f in feedback
            )
        folder = ctx.workspace.mkdir("work")
        result = run_tool_agent(
            ctx, task, workspace=folder, allow=cfg.allow, max_turns=cfg.max_turns, profile=cfg.profile, name="tool-task"
        )
        if result.stopped:
            raise StageFailed(
                f"the tool agent stopped ({result.stopped}) before finishing; see {result.transcript}", needs_human=True
            )
        files = sorted(ctx.relative(p) for p in folder.rglob("*") if p.is_file() and ".agent" not in p.parts)
        ctx.outputs.put(
            "tool_task_result",
            ToolTaskResult(task=task, reply=result.text, files=files, turns=result.turns, tool_calls=result.tool_calls),
        )
