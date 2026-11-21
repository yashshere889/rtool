"""`rtool validity submit` and `score`: cases to runs to a card, as on Barkla, with the demo model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from research_tool.cli.main import app
from research_tool.core.engine import Outcome
from research_tool.core.run import RunHandle
from research_tool.testing.demo_run import CLUSTER, advance, demo_case_data, demo_services
from research_tool.validity import provenance
from research_tool.validity.provenance import CodeVersion

CLEAN = CodeVersion(code="ab" * 32, commit="c" * 40)


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setattr(provenance, "code_version", lambda root=provenance.PACKAGE: CLEAN)
    cases = tmp_path / "cases" / "generic" / "phase-locked-recall"
    cases.mkdir(parents=True)
    case = demo_case_data(confirmed_by="demo model: its experiment simulates the effect")
    (cases / "case.yaml").write_text(yaml.safe_dump(case), encoding="utf-8")
    config = tmp_path / "cluster.yaml"
    config.write_text(yaml.safe_dump(CLUSTER), encoding="utf-8")
    return {"cases": tmp_path / "cases", "runs": tmp_path / "runs", "config": config}


def rtool(*args: str | Path) -> tuple[int, str]:
    result = CliRunner().invoke(app, [str(a) for a in args])
    return result.exit_code, result.output


def test_submit_then_score_writes_the_card(setup: dict[str, Path]) -> None:
    common = ("--runs-root", setup["runs"], "--config", setup["config"])
    code, out = rtool("validity", "submit", "generic", "--cases", setup["cases"], "--max-gpu-hours", "40", *common)
    assert code == 0, out
    assert "created  phase-locked-recall-main-r1-abababa" in out and "no real Slurm here" in out
    code, out = rtool("validity", "submit", "generic", "--cases", setup["cases"], *common)
    assert code == 0 and "exists   phase-locked-recall-main-r1-abababa" in out and "0 new run(s)" in out

    code, out = rtool("validity", "score", "generic", *common)
    assert code == 0 and "still running, not scored: 1" in out  # nothing has run yet

    run = RunHandle(setup["runs"] / "validity-generic" / "phase-locked-recall-main-r1-abababa")
    assert json.loads((run.root / "validity.json").read_text())["provenance"]["commit"] == "c" * 40
    assert advance(run, demo_services(run), max_episodes=60).outcome is Outcome.DONE

    code, out = rtool("validity", "score", "generic", *common)
    assert code == 0, out
    card = json.loads((setup["runs"] / ".validity-cards" / "generic-abababababab.json").read_text())
    assert card["headline"]["planted"]["k"] == card["headline"]["planted"]["n"] == 1
    assert card["provenance"]["model"] == "Qwen/Qwen3.6-35B-A3B" and card["provenance"]["versions"] == ["ab" * 32]
    assert card["cost"]["llm_calls"] > 0 and card["unfinished"] == []
    text = (setup["runs"] / ".validity-cards" / "generic-abababababab.md").read_text()
    assert "**planted**: 1 / 1 (100 %" in text and "commit `cccccccccc`" in text


def test_submit_refuses_uncommitted_code_and_unknown_cases(
    setup: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    common = ("--runs-root", setup["runs"], "--cases", setup["cases"])
    code, out = rtool("validity", "submit", "generic", "--case", "nope", *common)
    assert code != 0 and "no such case" in out
    monkeypatch.setattr(
        provenance, "code_version", lambda root=provenance.PACKAGE: CLEAN.model_copy(update={"dirty": True})
    )
    code, out = rtool("validity", "submit", "generic", *common)
    assert code != 0 and "changes git has not committed" in out
    assert not (setup["runs"] / "validity-generic").exists()
    code, out = rtool("validity", "score", "generic", "--runs-root", setup["runs"])
    assert code != 0 and "no validity runs of generic" in out
