from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from research_tool.agents.base import Agent, AgentContext, artifact_type, register_agent
from research_tool.llm.mock_backend import MockLLM
from research_tool.testing.toy_agents import ToyNumbers


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM().add("numbers about", ToyNumbers(values=[1.0, 2.0, 3.0, 4.0], source="mock"))


@pytest.fixture(autouse=True)
def _isolated_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a developer's ~/.config/rtool/config.yaml leak into tests."""
    empty = tmp_path / "empty-config.yaml"
    empty.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("RTOOL_CONFIG", str(empty))
    monkeypatch.setenv("RTOOL_SECRETS", str(tmp_path / "no-secrets.env"))  # nor real API keys


# ---------------------------------------------------------------- test agents
@artifact_type("test_value")
class ValueRecord(BaseModel):
    value: int


@register_agent("test_crashy")
class Crashy(Agent):
    """Stages its output, checkpoints, then 'dies' once (SystemExit) — simulates a killed episode."""

    consumes = ("research_brief",)
    produces = ("test_value",)
    needs_llm = False
    runs = 0

    def run(self, ctx: AgentContext) -> None:
        type(self).runs += 1
        if not ctx.state.get("staged"):
            ctx.outputs.put("test_value", ValueRecord(value=42))
            ctx.state["staged"] = True
            ctx.checkpoint()
            raise SystemExit("simulated node failure")


@register_agent("test_counter")
class Counter(Agent):
    """Counts how often run() is called (class attribute)."""

    consumes = ("research_brief",)
    produces = ("test_value",)
    needs_llm = False
    runs = 0

    def run(self, ctx: AgentContext) -> None:
        type(self).runs += 1
        ctx.outputs.put("test_value", ValueRecord(value=type(self).runs))


@register_agent("test_bad_requester")
class BadRequester(Agent):
    """Requests a change from a stage that is not upstream (must be ignored)."""

    consumes = ("test_value",)
    produces = ("toy_report",)
    needs_llm = False

    def run(self, ctx: AgentContext) -> None:
        from research_tool.testing.toy_agents import ToyReport

        ctx.request_change(stage="sibling", reason="not my upstream")
        ctx.outputs.put("toy_report", ToyReport(text="ok"))


@register_agent("test_report_critic")
class ReportCritic(Agent):
    """Reads the report; on its first attempt asks the *grand-parent* stage 'analyze' for the variance."""

    consumes = ("toy_report",)
    produces = ("test_value",)
    needs_llm = False

    def run(self, ctx: AgentContext) -> None:
        if ctx.attempt == 1:
            ctx.request_change(stage="analyze", reason="please include the variance")
        ctx.outputs.put("test_value", ValueRecord(value=ctx.attempt))


@register_agent("test_lazy")
class Lazy(Agent):
    """Declares an output but never produces it."""

    consumes = ("research_brief",)
    produces = ("test_value",)
    needs_llm = False

    def run(self, ctx: AgentContext) -> None:
        return None
