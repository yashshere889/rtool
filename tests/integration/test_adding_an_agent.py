"""docs/adding_an_agent.md must work as written: run its code blocks and its test."""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[2] / "docs" / "adding_an_agent.md"


def blocks(file_name: str) -> str:
    """The doc's ```python blocks that start with '# file: <file_name>', joined."""
    found = re.findall(r"```python\n# file: (\S+)\n(.*?)```", DOC.read_text(encoding="utf-8"), re.S)
    return "\n".join(code for name, code in found if name == file_name)


def test_the_documented_agent_and_its_test_run(runs_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("my_agents.novelty")
    monkeypatch.setitem(sys.modules, "my_agents.novelty", module)
    exec(compile(blocks("my_agents/novelty.py"), "my_agents/novelty.py", "exec"), module.__dict__)  # noqa: S102
    namespace: dict[str, object] = {"__name__": "doc_test"}
    exec(compile(blocks("tests/test_novelty.py"), "tests/test_novelty.py", "exec"), namespace)  # noqa: S102
    namespace["test_novelty_checker"](runs_root)  # type: ignore[operator]
