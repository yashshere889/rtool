from __future__ import annotations

import asyncio

from codeagent_helpers import call, run

from research_tool.codeagent import (
    ApprovalResponse,
    Hooks,
    PermissionEngine,
    PreToolUseOutcome,
    Tool,
    ToolCall,
    ToolContext,
    ToolExecutor,
    ToolInput,
    ToolRegistry,
)
from research_tool.codeagent.executor import partition


class SleepInput(ToolInput):
    seconds: float = 0.05


class Probe:
    def __init__(self) -> None:
        self.running = 0
        self.peak = 0
        self.order: list[str] = []


def make_sleep_tool(tool_name: str, safe: bool, probe: Probe, timeout: float | None = None) -> Tool:
    class SleepTool(Tool[SleepInput]):
        name = tool_name
        description = "sleep"
        Input = SleepInput
        timeout_s = timeout

        def is_read_only(self, inp):
            return safe

        def is_concurrency_safe(self, inp):
            return safe

        async def call(self, inp, ctx):
            probe.running += 1
            probe.peak = max(probe.peak, probe.running)
            probe.order.append(f"start:{self.name}")
            try:
                await asyncio.sleep(inp.seconds)
            finally:
                probe.running -= 1
            probe.order.append(f"end:{self.name}")
            return "ok"

    return SleepTool()


def test_partition_groups_consecutive_safe_calls():
    assert partition([True, True, False, True]) == [(True, [0, 1]), (False, [2]), (True, [3])]
    assert partition([False, False]) == [(False, [0]), (False, [1])]


def test_safe_calls_run_in_parallel_unsafe_alone(ctx):
    probe = Probe()
    registry = ToolRegistry([make_sleep_tool("fast_read", True, probe), make_sleep_tool("write", False, probe)])
    ex = ToolExecutor(registry, PermissionEngine(mode="bypass"))
    calls = [call("fast_read", 0), call("fast_read", 1), call("fast_read", 2), call("write", 3), call("fast_read", 4)]
    outcomes = run(ex.run(calls, ctx))
    assert [o.call.id for o in outcomes] == [c.id for c in calls]  # order preserved
    assert all(not o.result.is_error for o in outcomes)
    assert probe.peak == 3  # the three reads overlapped
    i = probe.order.index("start:write")
    assert probe.order[i + 1] == "end:write"  # nothing ran during the write


def test_unknown_tool_bad_json_and_bad_input_become_error_results(make_executor, ctx):
    ex = make_executor()
    calls = [
        call("raed_file", 0, path="x"),
        ToolCall(id="c1", name="read_file", arguments="{not json"),
        call("read_file", 2, path="README.md", limit="lots"),
        call("read_file", 3, path="README.md", bogus=True),
    ]
    out = run(ex.run(calls, ctx))
    assert all(o.result.is_error for o in out)
    assert "Did you mean 'read_file'" in out[0].result.content
    assert "JSON" in out[1].result.content
    assert "limit" in out[2].result.content
    assert "bogus" in out[3].result.content


def test_double_encoded_arguments_are_accepted(make_executor, ctx):
    import json

    ex = make_executor()
    raw = json.dumps(json.dumps({"path": "README.md"}))
    out = run(ex.run([ToolCall(id="c", name="read_file", arguments=raw)], ctx))
    assert not out[0].result.is_error and "# demo" in out[0].result.content


def test_timeout_and_cancellation(ctx):
    probe = Probe()
    registry = ToolRegistry([make_sleep_tool("slow", False, probe, timeout=0.05)])
    ex = ToolExecutor(registry, PermissionEngine(mode="bypass"))
    out = run(ex.run([call("slow", seconds=5)], ctx))
    assert out[0].result.is_error and "timed out" in out[0].result.content

    registry = ToolRegistry([make_sleep_tool("slow", False, probe)])
    ex = ToolExecutor(registry, PermissionEngine(mode="bypass"))

    async def interrupt():
        task = asyncio.create_task(ex.run([call("slow", seconds=5), call("slow", 1, seconds=5)], ctx))
        await asyncio.sleep(0.05)
        ctx.cancel.set()
        return await task

    out = run(interrupt())
    assert "Interrupted" in out[0].result.content
    assert "Cancelled" in out[1].result.content
    assert probe.running == 0


def test_large_output_is_saved_and_previewed(ctx):
    class BigInput(ToolInput):
        pass

    class Big(Tool[BigInput]):
        name = "big"
        description = "big"
        Input = BigInput
        max_result_chars = 1000

        def is_read_only(self, inp):
            return True

        async def call(self, inp, ctx):
            return "".join(f"line {i}\n" for i in range(2000))

    ex = ToolExecutor(ToolRegistry([Big()]), PermissionEngine())
    out = run(ex.run([call("big")], ctx))[0]
    path = out.result.metadata["full_output_path"]
    assert len(out.result.content) < 1500 and "characters omitted" in out.result.content
    assert open(path).read().count("\n") == 2000


def test_hooks_block_and_rewrite(make_executor, ctx, workspace):
    hooks = Hooks()

    @hooks.pre_tool_use("bash")
    def guard(tool, inp, ctx):
        if "rm " in inp["command"]:
            return PreToolUseOutcome(block="no deleting files")
        return PreToolUseOutcome(updated_input={**inp, "command": inp["command"].replace("Hello", "Bye")})

    seen = []

    @hooks.post_tool_use("*")
    async def record(tool, inp, result, ctx):
        seen.append((tool.name, result.is_error))

    ex = make_executor()
    ex.hooks = hooks
    ex.permissions = PermissionEngine(mode="bypass")
    out = run(ex.run([call("bash", 0, command="rm -rf src"), call("bash", 1, command="echo Hello")], ctx))
    assert "no deleting files" in out[0].result.content and (workspace / "src").exists()
    assert out[1].result.content == "Bye"
    assert seen == [("bash", False)]  # post hooks only run for calls that executed


def test_deferred_tool_must_be_loaded_with_tool_search(ctx):
    class PingInput(ToolInput):
        pass

    class Ping(Tool[PingInput]):
        name = "ping_service"
        description = "Check whether the staging service answers"
        search_hint = "health check staging"
        Input = PingInput
        defer_loading = True

        def is_read_only(self, inp):
            return True

        async def call(self, inp, ctx):
            return "pong"

    registry = ToolRegistry([Ping()])
    ex = ToolExecutor(registry, PermissionEngine())
    names = [t["function"]["name"] for t in registry.openai_tools(ctx)]
    assert names == ["tool_search"] and "ping_service" in registry.openai_tools(ctx)[0]["function"]["description"]

    first = run(ex.run([call("ping_service")], ctx))[0]
    assert first.result.is_error and "tool_search" in first.result.content

    found = run(ex.run([call("tool_search", query="staging health")], ctx))[0]
    assert "ping_service" in found.result.content
    assert [t["function"]["name"] for t in registry.openai_tools(ctx)] == ["ping_service", "tool_search"]
    assert run(ex.run([call("ping_service")], ctx))[0].result.content == "pong"


def test_parallel_approval_prompts_are_serialized(make_executor, ctx):
    active = 0
    peak = 0

    async def approver(req):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return ApprovalResponse(approved=True)

    ex = make_executor(approver=approver)
    outside = ToolContext(workspace=ctx.workspace / "src")  # README is outside this workspace -> ask
    calls = [call("read_file", i, path="../README.md") for i in range(3)]
    out = run(ex.run(calls, outside))
    assert all(not o.result.is_error for o in out)
    assert peak == 1
