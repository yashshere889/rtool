from __future__ import annotations

from codeagent_helpers import ScriptedModel, call, run, turn

from research_tool.codeagent import Agent, PermissionEngine, ToolExecutor, ToolRegistry, default_tools


def test_agent_reads_edits_and_finishes(ctx, workspace):
    model = ScriptedModel(
        [
            turn("Let me look.", [call("read_file", 0, path="src/app.py"), call("grep", 1, pattern="def add")]),
            turn("", [call("edit_file", 2, path="src/app.py", old_string="return a - b", new_string="return a + b")]),
            turn("Fixed add() to return a + b."),
        ]
    )
    registry = ToolRegistry(default_tools())
    executor = ToolExecutor(registry, PermissionEngine(mode="accept_edits"))
    events = []
    agent = Agent(model, registry, executor, ctx, on_event=events.append)

    answer = run(agent.send("add() is broken, fix it"))

    assert answer == "Fixed add() to return a + b."
    assert "return a + b" in (workspace / "src" / "app.py").read_text()

    # The second request carries both tool results, matched to their call ids, in call order.
    second_messages, tools = model.requests[1]
    tool_msgs = [m for m in second_messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_read_file_0", "call_grep_1"]
    assert "return a - b" in tool_msgs[0]["content"]
    assert {t["function"]["name"] for t in tools} == {"read_file", "edit_file", "write_file", "glob", "grep", "bash"}


def test_tool_errors_are_sent_back_not_raised(ctx):
    model = ScriptedModel(
        [turn("", [call("edit_file", path="src/app.py", old_string="x", new_string="y")]), turn("ok")]
    )
    registry = ToolRegistry(default_tools())
    agent = Agent(model, registry, ToolExecutor(registry, PermissionEngine()), ctx)
    assert run(agent.send("go")) == "ok"
    tool_msg = model.requests[1][0][-1]
    assert tool_msg["role"] == "tool" and tool_msg["content"].startswith("Error: Read src/app.py")


def test_schemas_are_flat(ctx):
    registry = ToolRegistry(default_tools())
    by_name = {t["function"]["name"]: t["function"]["parameters"] for t in registry.openai_tools(ctx)}
    glob = by_name["glob"]
    assert "title" not in glob and glob["properties"]["path"]["type"] == "string"
    assert glob["required"] == ["pattern"] and glob["additionalProperties"] is False


def test_cancel_during_model_request(ctx):
    import asyncio

    class SlowModel:
        async def complete(self, messages, tools):
            await asyncio.sleep(10)

    registry = ToolRegistry(default_tools())
    agent = Agent(SlowModel(), registry, ToolExecutor(registry, PermissionEngine()), ctx)

    async def go():
        task = asyncio.create_task(agent.send("hi"))
        await asyncio.sleep(0.05)
        ctx.cancel.set()
        return await asyncio.wait_for(task, 1)

    assert run(go()) == "[interrupted]"
