"""The vendored codeagent loop driven through rtool's LLM client: a stage, a budget, a refused command."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from research_tool.agents.base import Agent, AgentContext, register_agent
from research_tool.agents.tool_agent import ToolAgentResult
from research_tool.core.artifacts import ArtifactStore
from research_tool.core.engine import Engine, Outcome
from research_tool.core.events import EventLog
from research_tool.llm.mock_backend import MockLLM
from research_tool.llm.types import ChatRequest, ChatResponse
from research_tool.testing.harness import make_run, make_services

PIPELINE = {
    "name": "tool-task-only",
    "initial_artifacts": ["research_brief"],
    "stages": [{"id": "notes", "agent": "tool_task", "config": {"task": "Write notes.md about {topic}."}}],
}


BODY: list[Callable[[AgentContext], None]] = []


@register_agent("tool_agent_probe")
class Probe(Agent):
    """Runs whatever the test put in BODY, with a real stage context."""

    def run(self, ctx: AgentContext) -> None:
        BODY.pop()(ctx)


def run_inline(runs_root: Path, llm: MockLLM, body: Callable[[AgentContext], None]) -> None:
    pipeline = {
        "name": "probe",
        "initial_artifacts": ["research_brief"],
        "stages": [{"id": "p", "agent": "tool_agent_probe"}],
    }
    run = make_run(runs_root, pipeline)
    BODY.append(body)
    result = Engine(run, make_services(run, llm)).advance()
    assert result.outcome is Outcome.DONE, result.failed


def call(name: str, i: int, **args: object) -> dict:
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def scripted(*turns: ChatResponse) -> MockLLM:
    """Plays the turns back in order; records each request (tools, history) in ``llm.calls``."""
    queue = list(turns)
    return MockLLM([(lambda r: r.tools is not None, lambda r: queue.pop(0))])


def test_tool_task_stage_writes_files_through_tool_calls(runs_root: Path) -> None:
    run = make_run(runs_root, PIPELINE, topic="sleep spindles")
    llm = scripted(
        ChatResponse(text="", tool_calls=[call("write_file", 0, path="notes.md", content="# Sleep spindles\n")]),
        ChatResponse(text="", tool_calls=[call("bash", 1, command="curl https://example.org")]),
        ChatResponse(text="Wrote notes.md."),
    )

    result = Engine(run, make_services(run, llm)).advance()

    assert result.outcome is Outcome.DONE
    out = ArtifactStore(run.artifacts_dir).get("tool_task_result")
    assert out.reply == "Wrote notes.md." and "sleep spindles" in out.task
    assert out.tool_calls == {"write_file": 1, "bash": 1}
    (notes,) = [f for f in out.files if f.endswith("notes.md")]
    assert (run.root / notes).read_text() == "# Sleep spindles\n"
    # the request carried the tool schemas, and the refused curl reached the model as an error it can act on
    first, third = llm.calls[0], llm.calls[2]
    assert {t["function"]["name"] for t in first.tools} >= {"read_file", "write_file", "bash"}
    refused = third.messages[-1]
    assert refused.role == "tool" and refused.tool_call_id == "c1" and "deny rule" in refused.content
    (event,) = EventLog(run.events_path, run.run_id).read(kinds=["tool_agent"])
    assert event["data"]["turns"] == 3 and event["data"]["stopped"] is None


def test_the_same_refused_call_five_times_stops_the_stage(runs_root: Path) -> None:
    run = make_run(runs_root, PIPELINE)
    again = ChatResponse(text="", tool_calls=[call("bash", 0, command="python run.py")])  # not allowed by default
    llm = MockLLM([(lambda r: isinstance(r, ChatRequest), again)])

    result = Engine(run, make_services(run, llm)).advance()

    assert result.outcome is not Outcome.DONE
    assert len(llm.calls) == 5  # stopped by the fifth refusal, not by max_turns (40)
    (event,) = EventLog(run.events_path, run.run_id).read(kinds=["tool_agent"])
    assert event["data"]["stopped"] == "repeated_denial" and event["data"]["denials"] == 5


def test_read_roots_open_reads_outside_the_workspace_and_nothing_else(runs_root: Path, tmp_path: Path) -> None:
    from research_tool.agents.tool_agent import run_tool_agent

    data, secret = tmp_path / "data", tmp_path / "secret"
    for folder in (data, secret):
        folder.mkdir()
        (folder / "f.txt").write_text(f"in {folder.name}\n")
    turns = [
        [call("read_file", 0, path=str(data / "f.txt")), call("read_file", 1, path=str(secret / "f.txt"))],
        [],
    ]
    llm = MockLLM([(lambda r: r.tools is not None, lambda r: ChatResponse(text="ok", tool_calls=turns.pop(0)))])
    seen: list[ToolAgentResult] = []

    def body(ctx: AgentContext) -> None:
        seen.append(run_tool_agent(ctx, "read both", workspace=ctx.workspace.mkdir("work"), read_roots=[data]))

    run_inline(runs_root, llm, body)

    results = [m.content for m in llm.calls[-1].messages if m.role == "tool"]
    assert "in data" in results[0] and "Error:" in results[1] and "in secret" not in results[1]
    assert seen[0].denials and "secret" in seen[0].denials[0]
