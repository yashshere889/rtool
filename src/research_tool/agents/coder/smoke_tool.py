"""The Coder's own tool for the tool-calling loop: a smoke test through rtool's harness and experiment image.

The episode node's Python is not the experiment environment, so the loop gets no ``bash(python *)``;
it tests the code the same way rtool's smoke phase does (``ExperimentEnv``: container, packages, binds).
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from pydantic import Field

from ...codeagent import PermissionDecision, Tool, ToolContext, ToolError, ToolInput, ToolResult
from .results import read_output

if TYPE_CHECKING:
    from .agent import CoderSession


class SmokeInput(ToolInput):
    condition: str = Field(description="One of the plan's conditions")
    seed: int = Field(0, description="Random seed passed to run()")


class SmokeTestTool(Tool[SmokeInput]):
    name = "smoke_test"
    Input = SmokeInput

    def __init__(self, session: CoderSession) -> None:
        self.session = session

    @property
    def description(self) -> str:  # type: ignore[override]
        return (
            "Run experiment.py in smoke mode exactly as rtool will check it: `python run.py --condition C "
            "--seed S --smoke` in the experiment environment (its packages and data). Returns the metrics, or "
            f"what is wrong with the output, and the log tail. Conditions: {', '.join(self.session.conditions)}."
        )

    async def check_permissions(self, inp: SmokeInput, ctx: ToolContext) -> PermissionDecision:
        return PermissionDecision.allow("rtool's own harness")

    async def call(self, inp: SmokeInput, ctx: ToolContext) -> ToolResult | str:
        s = self.session
        if inp.condition not in s.conditions:
            raise ToolError(f"unknown condition {inp.condition!r}; use one of {s.conditions}")
        if not (s.code / "experiment.py").exists():
            raise ToolError("experiment.py does not exist yet; write it first")
        name = f"{inp.condition}__{inp.seed}"
        installed = await asyncio.to_thread(s.env.install, s.requirements(), log_path=s.logs / "agent-install.log")
        if installed is not None and not installed.ok:
            return ToolResult.error(f"pip could not install {s.requirements()}:\n{installed.tail(40)}")
        out = s.scratch / "agent-smoke" / f"{name}.json"
        out.unlink(missing_ok=True)
        args = ["run.py", "--condition", inp.condition, "--seed", str(inp.seed), "--output", str(out), "--smoke",
                "--workdir", str(s.scratch / "agent-smoke" / name)]  # fmt: skip
        timeout = s.ctx.app_config.experiments.smoke_timeout_s
        run = await asyncio.to_thread(s.env.run, args, log_path=s.logs / f"agent-smoke-{name}.log", timeout_s=timeout)
        data, problem = read_output(out, s.metric_names)
        if run.ok and data is not None:
            return f"passed in {run.seconds:.1f} s; metrics {json.dumps(data['metrics'])}\n\nlog tail:\n{run.tail(30)}"
        status = f"timed out after {timeout:g} s" if run.timed_out else f"exit code {run.exit_code}"
        return ToolResult.error(f"failed ({status}; {problem})\n\nlog tail:\n{run.tail(80)}")
