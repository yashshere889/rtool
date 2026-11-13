"""`rtool replay`: a copy of a run redoes one stage, under an overlay config, without touching the source."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_tool.core.engine import Engine
from research_tool.core.errors import RToolError
from research_tool.core.run import RunHandle
from research_tool.core.state import StageStatus, StateStore
from research_tool.llm.mock_backend import MockLLM
from research_tool.service import replay_run
from research_tool.testing.harness import drive, make_run, make_services


def test_replay_redoes_one_stage_under_an_overlay(runs_root: Path, mock_llm: MockLLM, tmp_path: Path) -> None:
    source = make_run(runs_root, "toy")
    drive(source, make_services(source, mock_llm))  # stops at the numbers_review gate
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("llm: {served_model: qwen-test}\nslurm: {episode: {gpus: 2}}\n")

    copy = replay_run(source, "source", "qwen", [overlay])

    assert copy.run_id == f"{source.run_id}-source-qwen" and copy.root.parent == source.root.parent
    assert copy.config.llm.served_model == "qwen-test" and copy.config.slurm.episode.gpus == 2
    assert str(overlay.resolve()) in copy.config.slurm.episode.config_files  # episode jobs re-read it
    assert (copy.artifacts_dir / "research_brief").is_dir()  # same inputs
    assert not copy.events_path.exists()  # stats count only the replay
    assert StateStore(copy.state_dir).load_stage("source").status is StageStatus.STALE

    Engine(copy, make_services(copy, mock_llm)).advance()
    redone = StateStore(copy.state_dir).load_stage("source")
    assert redone.status is StageStatus.AWAITING_GATE and redone.attempt == 2  # ran again, stopped at its gate
    original = StateStore(RunHandle(source.root).state_dir).load_stage("source")
    assert original.attempt == 1 and original.status is StageStatus.AWAITING_GATE  # source untouched


def test_replay_refuses_unfinished_inputs_and_unknown_stages(runs_root: Path, mock_llm: MockLLM) -> None:
    source = make_run(runs_root, "toy")
    drive(source, make_services(source, mock_llm))
    with pytest.raises(RToolError, match="not all done"):
        replay_run(source, "analyze", "x")
    with pytest.raises(RToolError, match="no stage"):
        replay_run(source, "nope", "x")
    replay_run(source, "source", "x")
    with pytest.raises(RToolError, match="already exists"):
        replay_run(source, "source", "x")
