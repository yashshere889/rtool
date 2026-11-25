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


@pytest.fixture(scope="session")
def pipelines_demo_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The study and synthesis demos, kept apart from ``demo_root`` (tests list every run there)."""
    root = tmp_path_factory.mktemp("pipeline-demos")
    for pipeline, stops in (
        ("study", ("protocol_review", "data_upload", "final_report", "done")),
        ("synthesis", ("literature_review", "final_report", "done")),
    ):
        for stop_at in stops:
            build_demo_run(root, stop_at, pipeline=pipeline)
    return root


@pytest.fixture
def pipelines_root(pipelines_demo_root: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "runs"
    shutil.copytree(pipelines_demo_root, copy)
    return copy
