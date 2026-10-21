"""The real research pipeline end to end: literature → hypotheses → plan → fetch, through every gate.

Each agent is tested alone elsewhere; this checks that what one stage writes is
exactly what the next one can read (KB snapshot paths, verified quotes, the
selected hypothesis, the plan's resources).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx
from test_literature_agent import (
    HITS,
    Documents,
    FakeS2,
    FakeSearch,
    fake_pdf_to_text,
    make_llm,
)

from research_tool.agents.literature.toolkit import LiteratureToolkit
from research_tool.core.actions import ActionKind
from research_tool.core.artifacts import ArtifactStore
from research_tool.core.engine import Engine, Outcome
from research_tool.core.services import Services
from research_tool.hpc.fake_slurm import FakeSlurm
from research_tool.knowledge.vectors import HashingEmbedder
from research_tool.llm.client import LLMClient
from research_tool.llm.types import ChatRequest
from research_tool.testing.harness import make_run, queue
from research_tool.tools.http import PoliteHttp
from research_tool.tools.resources.checks import ResourceCheck
from research_tool.tools.resources.fetch import Fetcher


def schema(request: ChatRequest) -> str:
    return str((request.json_schema or {}).get("title", ""))


def drafts(request: ChatRequest) -> dict[str, Any]:
    """Cite the first evidence line verbatim, as an honest model would."""
    prompt = request.messages[-1].content
    paper_id, quote = re.search(r"^\[([^\]]+)\] (.+)$", prompt, re.M).groups()  # type: ignore[union-attr]
    n = len(re.findall(r"Opportunity", prompt))
    return {
        "hypotheses": [
            {
                "title": f"Hypothesis from {paper_id}",
                "statement": f"If spindles are enhanced in study {n} {paper_id} then recall improves",
                "null_hypothesis": "No change",
                "rationale": "Evidence links spindles to recall",
                "evidence": [{"paper_id": paper_id, "quote": quote}],
                "independent_variables": ["stimulation"],
                "dependent_variables": ["recall"],
                "datasets": ["Sleep-EDF"],
                "methods": ["EEG"],
                "success_criteria": "recall up",
                "compute": "small_gpu",
            }
        ]
    }


def add_downstream_rules(llm: Any) -> None:
    llm.add(lambda r: schema(r) == "HypothesisDrafts", drafts)
    llm.add(
        lambda r: schema(r) == "NoveltyJudgment",
        {"novel": True, "closest_paper_id": "", "overlap": "new population", "score": 8},
    )
    llm.add(
        lambda r: schema(r) == "RubricScores",
        {"novelty": 4, "feasibility": 4, "testability": 4, "significance": 4, "clarity": 4, "comment": "ok"},
    )
    llm.add(
        lambda r: schema(r) == "PlanCore",
        {
            "objective": "Test stimulation",
            "design": "Compare stimulation with sham",
            "independent_variables": [{"name": "stimulation", "description": "on/off"}],
            "dependent_variables": [{"name": "recall", "description": "recall gain"}],
            "conditions": [
                {"name": "sham", "description": "no sound", "is_baseline": True},
                {"name": "stim", "description": "closed-loop sound"},
            ],
            "metrics": [{"name": "recall_gain", "definition": "post minus pre"}],
            "statistical_tests": [{"test": "paired t-test", "comparison": "stim vs sham"}],
            "n_seeds": 3,
            "procedure": ["load EEG", "detect spindles", "simulate stimulation", "score recall"],
            "expected_results": "stim better",
        },
    )
    llm.add(
        lambda r: schema(r) == "ResourcePlan",
        {
            "resources": [
                {
                    "kind": "dataset",
                    "name": "Sleep-EDF",
                    "source": "huggingface",
                    "identifier": "physionet/sleep-edf",
                    "purpose": "EEG recordings",
                }
            ]
        },
    )
    llm.add(
        lambda r: schema(r) == "ComputeEstimate",
        {"gpu_memory_gb": 8, "gpus_per_run": 1, "hours_per_run": 1, "justification": "small model"},
    )


EXPERIMENT = '''"""Closed-loop stimulation, simulated."""
# requirements:
import random


def run(condition, seed, resources, smoke, workdir):
    rng = random.Random(seed)
    gain = (0.10 if condition == "sham" else 0.18) + rng.uniform(-0.01, 0.01)
    return {"recall_gain": gain}
'''


def section(request: ChatRequest) -> str:
    """Prose that follows the rules: a known citation key and a listed placeholder."""
    prompt = request.messages[-1].content
    keys = [k for k in re.findall(r"@([A-Za-z0-9_:.\-]+)", prompt) if k not in ("key", "a", "b")]
    placeholder = re.search(r"^- (\{\{[a-z_]+:recall_gain:stim\}\})", prompt, re.M)
    text = "Sleep spindles are linked to memory consolidation"
    text += f" [@{keys[0]}]." if keys else "."
    if placeholder:
        text += f" With stimulation the recall gain reached {placeholder.group(1)}."
    if '"Discussion"' in prompt:
        text += "\n\nLimitations. Only a few seeds were run."
    return text


def add_report_rules(llm: Any) -> None:
    llm.add(
        lambda r: "Write the complete Python file `experiment.py`" in r.messages[-1].content,
        f"```python\n{EXPERIMENT}```",
    )
    llm.add(
        lambda r: "You are editing the" in r.messages[-1].content,
        lambda r: section(r) + " Conditions were compared with a paired t-test.",
    )
    llm.add(lambda r: "section of a research report" in r.messages[-1].content, section)
    llm.add(
        lambda r: schema(r) == "TitleAbstract",
        {"title": "Closed-loop stimulation and recall", "abstract": "Across {{runs_ok}} runs, recall improved."},
    )
    llm.add(lambda r: schema(r) == "Contradictions", {"items": []})
    llm.add(
        lambda r: schema(r) == "ReviewDraft",
        {
            "scores": {"soundness": 4, "evidence": 4, "clarity": 4, "reproducibility": 5, "limitations": 4},
            "summary": "A small but careful study.",
            "issues": [
                {
                    "section": "results",
                    "problem": "The text does not name the statistical test",
                    "suggestion": "Say that a paired t-test was used",
                    "target": "report",
                    "severity": "minor",
                }
            ],
        },
    )


class Checker:
    def check(self, resource: Any) -> ResourceCheck:
        return ResourceCheck(True, license="odc-by", size_gb=1.2)


def snapshot(repo_id: str, repo_type: str, local_dir: str, token: str | None) -> str:
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    (Path(local_dir) / "recording.edf").write_bytes(b"0" * 64)
    return local_dir


def test_topic_to_reviewed_report(runs_root: Path, tmp_path: Path) -> None:
    run = make_run(
        runs_root,
        "research",
        topic="Sleep spindles and memory consolidation",
        overrides={"paths": {"resources_root": str(tmp_path / "resources")}},
    )
    calls: dict[str, int] = {}
    llm = make_llm(calls)
    add_downstream_rules(llm)
    add_report_rules(llm)
    documents = Documents()

    def toolkit(config: Any, cache_path: Path) -> LiteratureToolkit:
        http = PoliteHttp(
            client=httpx.Client(transport=httpx.MockTransport(documents)), sleep=lambda _s: None, default_interval_s=0.0
        )
        return LiteratureToolkit(
            sources={"openalex": FakeSearch(HITS), "semantic_scholar": FakeS2([])},
            unpaywall=None,
            http=http,
            pdf_converter=fake_pdf_to_text,
            embedder=HashingEmbedder(),
        )

    slurm = FakeSlurm(run.jobs_dir / "fake_slurm.json", run_commands=True)
    services = Services(
        llm=LLMClient(llm, run.config.llm),
        slurm=slurm,
        tools={
            "literature_toolkit": toolkit,
            "resource_checker": Checker(),
            "resource_fetcher": lambda root: Fetcher(root, snapshot=snapshot),
        },
    )

    def step(expected_gate: str | None) -> Any:
        result = Engine(run, services).advance()
        if expected_gate is None:
            assert result.outcome is Outcome.DONE, result.message
        else:
            assert result.outcome is Outcome.WAITING_GATE, (result.message, result.failed)
            assert result.open_gates[0]["gate"] == expected_gate
        return result

    step("literature_review")
    queue(run, ActionKind.APPROVE)
    selection = step("hypothesis_selection")
    assert selection.open_gates[0]["summary"]["options"]
    queue(run, ActionKind.SELECT, payload={"index": 0})
    step("experiment_plan")
    queue(run, ActionKind.APPROVE)
    step("gpu_runs")  # resources fetched, code written and smoke-tested; now the GPU spend needs a yes
    queue(run, ActionKind.APPROVE)
    waiting = Engine(run, services).advance()
    assert waiting.outcome is Outcome.WAITING_JOB, waiting.message
    slurm.finish_all()  # the six experiment runs, executed for real by the fake cluster
    final = step("final_report")
    assert any(path.endswith("final_report.md") for path in final.open_gates[0]["summary"]["review_files"])
    queue(run, ActionKind.APPROVE)
    step(None)

    store = ArtifactStore(run.artifacts_dir)
    corpus, hypotheses = store.get("corpus"), store.get("hypothesis_set")
    chosen = store.get("selected_hypothesis")
    # the hypothesis is grounded in a quote the literature agent verified in a real downloaded paper
    assert chosen.item["evidence"][0]["verified"] and chosen.item["evidence"][0]["paper_id"] in {
        p.id for p in corpus.included()
    }
    assert hypotheses.hypotheses[0].title == chosen.item["title"]
    plan = store.get("experiment_plan")
    assert plan.hypothesis == chosen.item["statement"] and not plan.issues
    assert plan.primary_metric().name == "recall_gain" and plan.metrics[0].primary  # the only metric, by default
    manifest = store.get("resource_manifest")
    assert manifest.resources[0].status == "ready" and Path(manifest.resources[0].path).is_dir()
    # provenance: every artifact records which stage, agent and inputs produced it
    assert store.envelope("experiment_plan").inputs["selected_hypothesis"] == 1

    results = store.get("experiment_results")
    assert results.complete and len(results.runs) == 6
    stim = results.comparison("stim", "recall_gain")
    assert stim is not None and stim.significant and stim.better
    assert stim.analysis == "confirmatory" and stim.correction == "none"  # one comparison: nothing to correct
    report, final, review = store.get("report"), store.get("final_report"), store.get("review")
    text = (run.root / final.markdown_path).read_text()
    # numbers come from the results, citations from the corpus
    assert results.summary("stim", "recall_gain").mean is not None  # type: ignore[union-attr]
    assert "{{" not in text and report.citations and set(report.citations) <= {p.bib_key for p in corpus.papers}
    assert "## References" in text and f"**[{report.citations[0]}]**" in text
    assert "| `stim` vs `sham` |" in text and "figures/recall_gain.svg" in text
    assert (run.root / final.markdown_path).parent.joinpath("figures", "recall_gain.svg").exists()
    assert review.verdict == "accept", review.checks
    assert review.edited_sections == ["results"] and review.scores["soundness"] == 4
    assert all(check.ok for check in review.checks)
    assert "+" in (run.root / review.diff_path).read_text()  # the reviewer's edit, as a diff
