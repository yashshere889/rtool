"""Fixtures for the vendored codeagent tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_tool.codeagent import PermissionEngine, ToolContext, ToolExecutor, ToolRegistry, default_tools


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "README.md").write_text("# demo\n")
    return tmp_path


@pytest.fixture
def make_executor():
    def factory(tools=None, **perm_kwargs):
        registry = ToolRegistry(tools if tools is not None else default_tools())
        return ToolExecutor(registry, PermissionEngine(**perm_kwargs))

    return factory


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace)
