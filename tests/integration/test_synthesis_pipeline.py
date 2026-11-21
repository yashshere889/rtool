"""The literature-only pipeline: the demo's papers, a review whose every sentence rests on verified quotes.

The demo model answers the literature stage as usual (real PDFs of fictional papers, real quote checks);
for the review it restates each finding it is offered and adds two sentences that must be dropped.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from research_tool.core.artifacts import ArtifactStore, Producer
from research_tool.core.config import deep_merge, load_config
from research_tool.core.engine import Outcome
from research_tool.core.fs import atomic_write_json, read_json
from research_tool.core.run import RunHandle
from research_tool.llm.types import ChatRequest
from research_tool.testing import demo_data
from research_tool.testing.demo_run import CLUSTER, advance, demo_services
from research_tool.validity.cases import Case
from research_tool.validity.runs import MARKER, create_case_run
from research_tool.validity.score import score_run

CASE: dict[str, Any] = {
    "id": "stimulation-review",
    "kind": "synthesis",
    "pack": "generic",
    "confirmed_by": "the demo papers state these findings (testing/demo_data.py)",
    "brief": {"topic": demo_data.TOPIC},
    "expect": {
        "findings": [
            {"paper": "Open-loop versus closed-loop acoustic stimulation", "terms": ["phase-locked", "recall"]},
            {"paper": "Ageing weakens the response", "terms": ["older adults"]},
        ]
    },
}


def paragraph(request: ChatRequest) -> dict[str, Any]:
    offered = re.findall(r"^\[(C\d+)\] (.+?) \(quote:", request.messages[-1].content, re.M)
    sentences = [{"text": text, "claims": [cid]} for cid, text in offered]
    first = offered[0][0]
    sentences += [
        {"text": "Every study found an effect in 97 % of participants.", "claims": [first]},  # a number of its own
        {"text": "Stimulation also cures insomnia.", "claims": ["C999"]},  # a finding nobody offered
    ]
    return {"sentences": sentences[:14]}


@pytest.fixture(scope="module")
def reviewed(tmp_path_factory: pytest.TempPathFactory) -> RunHandle:
    root = tmp_path_factory.mktemp("synthesis") / "runs"
    config = load_config(overrides=deep_merge(CLUSTER, {"paths": {"runs_root": str(root)}}), env={})
    run = create_case_run(config, Case.model_validate(CASE), commit="0" * 12)
    services = demo_services(run)
    services.llm.backend.rules.insert(0, (lambda r: (r.json_schema or {}).get("title") == "ParagraphDraft", paragraph))
    result = advance(run, services, max_episodes=60)
    assert result.outcome is Outcome.DONE, f"{result.outcome.value}: {result.message}"
    return run


def test_the_review_keeps_only_sentences_its_quotes_support(reviewed: RunHandle) -> None:
    assert [s.id for s in reviewed.spec.pipeline.stages] == ["literature", "synthesis"]
    store = ArtifactStore(reviewed.artifacts_dir)
    review = store.get("synthesis")
    kept, dropped = review.kept(), review.dropped
    assert kept and all(s.claims and s.papers for s in kept)
    reasons = {d.reason.split(":")[0] for d in dropped}
    assert reasons == {"states numbers its findings do not", "cites findings that were not offered"}
    assert len(dropped) == 2 * len(review.sections)  # both bad sentences of every paragraph went
    markdown = (reviewed.root / store.get("final_report").markdown_path).read_text()
    assert "## Evidence" in markdown and "## References" in markdown and "[@" in markdown
    assert "97 %" not in markdown and "insomnia" not in markdown
    assert {c.name for c in review.checks} == {
        "sentences resting on verified findings",
        "findings used",
        "unsupported sentences",
    }


def test_the_review_states_the_known_findings(reviewed: RunHandle) -> None:
    score = score_run(reviewed)
    assert (score.level, score.category) == (3, ""), [c for c in score.checks if not c.ok]
    assert score.facts["recall"] == 1.0 and score.facts["dropped_sentences"] >= 2


def copy_of(run: RunHandle, tmp_path: Path) -> RunHandle:
    shutil.copytree(run.root, tmp_path / run.run_id)
    return RunHandle(tmp_path / run.run_id)


def test_a_missed_finding_and_an_unfaithful_sentence_are_caught(reviewed: RunHandle, tmp_path: Path) -> None:
    missing = copy_of(reviewed, tmp_path / "a")
    marker = read_json(missing.root / MARKER)
    case = deep_merge(CASE, {})
    case["expect"]["findings"] = [*CASE["expect"]["findings"], {"paper": "Phase-locked auditory", "terms": ["placebo"]}]
    atomic_write_json(missing.root / MARKER, {**marker, "case": Case.model_validate(case).model_dump(mode="json")})
    score = score_run(missing)
    assert (score.level, score.category) == (2, "missed_findings") and score.facts["recall"] == pytest.approx(
        2 / 3, 1e-3
    )

    edited = copy_of(reviewed, tmp_path / "b")
    store = ArtifactStore(edited.artifacts_dir)
    review = store.get("synthesis")
    review.sections[0].sentences[0].text += " It held in 42 of 50 nights."  # numbers no quote states
    store.put("synthesis", review, producer=Producer(stage="test", agent="test"))
    assert (score_run(edited).level, score_run(edited).category) == (1, "unfaithful_review")
