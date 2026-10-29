"""Demo runs for the web API tests, built once per session; each test gets its own copy."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from research_tool.testing.demo_run import build_demo_run  # noqa: E402


@pytest.fixture(scope="session")
def demo_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("demo-runs")
    for stop_at in ("literature_review", "hypothesis_selection", "experiment_plan", "gpu_runs", "final_report", "done"):
        build_demo_run(root, stop_at)
    return root


@pytest.fixture
def runs_root(demo_root: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "runs"
    shutil.copytree(demo_root, copy)
    return copy
