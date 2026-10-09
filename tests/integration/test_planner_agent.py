"""Planner + fetch_resources: repair round, compute fitting, plan gate, then downloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_tool.core.actions import ActionKind
from research_tool.core.artifacts import ArtifactStore, Producer
from research_tool.core.engine import Engine, Outcome
from research_tool.core.run import RunHandle
from research_tool.core.services import Services
from research_tool.knowledge.store import KnowledgeStore
from research_tool.llm.client import LLMClient
from research_tool.llm.mock_backend import MockLLM
from research_tool.llm.types import ChatRequest
from research_tool.schemas.common import SelectedItem
from research_tool.schemas.hypothesis import CheckedEvidence, Hypothesis
from research_tool.schemas.literature import Corpus, Entity, KnowledgeMap, PaperRecord
from research_tool.testing.harness import make_run, queue
from research_tool.tools.documents.sections import Chunk
from research_tool.tools.resources.checks import ResourceCheck
from research_tool.tools.resources.fetch import Fetcher

BARKLA = Path(__file__).resolve().parents[2] / "configs" / "barkla.yaml"
PIPELINE = {
    "name": "p",
    "initial_artifacts": ["research_brief", "selected_hypothesis", "corpus", "knowledge_map"],
    "gates": {"plan_gate": {"kind": "approve"}},
    "stages": [{"id": "plan", "agent": "planner", "gate": "plan_gate"}, {"id": "fetch", "agent": "fetch_resources"}],
}


def seed(runs_root: Path, tmp_path: Path) -> RunHandle:
    from research_tool.core.config import load_config

    site = load_config(BARKLA, env={"USER": "me"}).model_dump(mode="json")
    overrides = {"slurm": site["slurm"] | {"backend": "fake"}, "paths": {"resources_root": str(tmp_path / "res")}}
    run = make_run(runs_root, PIPELINE, topic="Sentiment robustness", overrides=overrides)
    paper = PaperRecord(id="p1", title="Robust sentiment models", status="extracted")
    with KnowledgeStore(run.root / "kb.sqlite") as kb:
        kb.upsert_papers([paper])
        kb.set_chunks("p1", [Chunk(0, "Methods", "We fine-tune BERT on IMDB reviews with three seeds.")])
    hypothesis = Hypothesis(
        id="h1",
        title="Noise hurts BERT",
        statement="If label noise is added then BERT sentiment accuracy drops",
        null_hypothesis="No drop",
        rationale="r",
        evidence=[CheckedEvidence(paper_id="p1", quote="q", verified=True, support=1)],
        independent_variables=["noise rate"],
        dependent_variables=["accuracy"],
        success_criteria="drop > 2 points",
        compute="small_gpu",
        datasets=["IMDB"],
        methods=["BERT"],
    )
    store = ArtifactStore(run.artifacts_dir)
    producer = Producer(stage="hypotheses", agent="test")
    store.put("selected_hypothesis", SelectedItem(index=0, item=hypothesis.model_dump()), producer=producer)
    store.put(
        "corpus",
        Corpus(topic="t", queries=[], papers=[paper], kb_path="kb.sqlite", kb_sha256="x", bibtex_path="refs.bib"),
        producer=producer,
    )
    store.put(
        "knowledge_map",
        KnowledgeMap(topic="t", overview="", entities=[Entity(kind="dataset", name="IMDB", paper_ids=["p1"])]),
        producer=producer,
    )
    return run


def schema(request: ChatRequest) -> str:
    return str((request.json_schema or {}).get("title", ""))


def core(request: ChatRequest) -> dict[str, Any]:
    fixed = "Problems with the previous version" in request.messages[-1].content
    return {
        "objective": "Measure accuracy under label noise",
        "design": "Fine-tune with and without noise",
        "independent_variables": [
            {"name": "noise rate", "description": "fraction of flipped labels", "values": ["0", "0.2"]}
        ],
        "dependent_variables": [{"name": "accuracy", "description": "test accuracy"}],
        "conditions": [
            {"name": "Clean Labels", "description": "no noise", "is_baseline": fixed},
            {"name": "Noisy Labels", "description": "20% noise"},
        ],
        "metrics": [{"name": "Accuracy", "definition": "test accuracy"}],
        "statistical_tests": [{"test": "paired t-test", "comparison": "clean vs noisy accuracy"}],
        "n_seeds": 3,
        "procedure": ["load IMDB", "flip labels", "fine-tune BERT", "evaluate"],
        "expected_results": "noisy is worse",
        "risks": ["small effect"],
    }


def resources(request: ChatRequest) -> dict[str, Any]:
    fixed = "Problems with the previous list" in request.messages[-1].content
    return {
        "resources": [
            {
                "kind": "dataset",
                "name": "IMDB",
                "source": "huggingface",
                "identifier": "stanfordnlp/imdb" if fixed else "made-up/imdb-v9",
                "purpose": "data",
            },
            {
                "kind": "model",
                "name": "BERT base",
                "source": "huggingface",
                "identifier": "google-bert/bert-base-uncased",
                "purpose": "model",
            },
            {
                "kind": "code",
                "name": "transformers",
                "source": "pip",
                "identifier": "transformers",
                "purpose": "training",
            },
        ]
    }


class Checker:
    def check(self, resource: Any) -> ResourceCheck:
        if resource.identifier.startswith("made-up/"):
            return ResourceCheck(False, note="HTTP 404")
        return ResourceCheck(True, license="apache-2.0", size_gb=0.5)


def fake_snapshot(repo_id: str, repo_type: str, local_dir: str, token: str | None) -> str:
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    (Path(local_dir) / "data.bin").write_bytes(b"x" * 10)
    return local_dir


def services(run: RunHandle, llm: MockLLM) -> Services:
    return Services(
        llm=LLMClient(llm, run.config.llm),
        tools={"resource_checker": Checker(), "resource_fetcher": lambda root: Fetcher(root, snapshot=fake_snapshot)},
    )


def test_plan_is_repaired_fitted_and_then_fetched(runs_root: Path, tmp_path: Path) -> None:
    run = seed(runs_root, tmp_path)
    llm = MockLLM()
    llm.add(lambda r: schema(r) == "PlanCore", core)
    llm.add(lambda r: schema(r) == "ResourcePlan", resources)
    llm.add(
        lambda r: schema(r) == "ComputeEstimate",
        {"gpu_memory_gb": 14, "gpus_per_run": 1, "hours_per_run": 1.5, "justification": "BERT base fits"},
    )
    result = Engine(run, services(run, llm)).advance()
    assert result.outcome is Outcome.WAITING_GATE and result.open_gates[0]["gate"] == "plan_gate"

    plan = ArtifactStore(run.artifacts_dir).get("experiment_plan")
    assert plan.issues == []  # the repair round fixed the missing baseline and the made-up dataset
    assert [c.name for c in plan.conditions] == ["clean_labels", "noisy_labels"] and plan.conditions[0].is_baseline
    assert plan.metrics[0].name == "accuracy" and plan.seeds == [0, 1, 2]
    assert plan.resources[0].identifier == "stanfordnlp/imdb" and all(r.verified for r in plan.resources)
    assert plan.compute.partition == "gpu-l40s" and plan.compute.runs == 6 and plan.compute.time_limit == "01:57:00"
    assert plan.results_schema["properties"]["runs"]["items"]["properties"]["condition"]["enum"] == [
        "clean_labels",
        "noisy_labels",
    ]
    asked = [schema(r) for r in llm.calls]
    assert asked.count("PlanCore") == 2 and asked.count("ResourcePlan") == 2 and asked.count("ComputeEstimate") == 1
    assert "experiment_plan.md" in result.open_gates[0]["summary"]["review_files"][0]

    queue(run, ActionKind.APPROVE)
    assert Engine(run, services(run, llm)).advance().outcome is Outcome.DONE
    manifest = ArtifactStore(run.artifacts_dir).get("resource_manifest")
    statuses = {r.name: r.status for r in manifest.resources}
    assert statuses == {"IMDB": "ready", "BERT base": "ready", "transformers": "deferred"}
    assert manifest.root == str(tmp_path / "res")


def test_unfixable_problems_reach_the_human(runs_root: Path, tmp_path: Path) -> None:
    run = seed(runs_root, tmp_path)
    llm = MockLLM()
    llm.add(lambda r: schema(r) == "PlanCore", core)
    llm.add(
        lambda r: schema(r) == "ResourcePlan",
        {
            "resources": [
                {
                    "kind": "dataset",
                    "name": "Ghost",
                    "source": "huggingface",
                    "identifier": "made-up/ghost",
                    "purpose": "x",
                }
            ]
        },
    )
    llm.add(
        lambda r: schema(r) == "ComputeEstimate",
        {"gpu_memory_gb": 14, "gpus_per_run": 1, "hours_per_run": 1, "justification": "x"},
    )
    Engine(run, services(run, llm)).advance()
    plan = ArtifactStore(run.artifacts_dir).get("experiment_plan")
    assert any("Ghost" in issue and "could not be found" in issue for issue in plan.issues)
    assert "Unresolved issues" in (run.root / plan.markdown_path).read_text()


def test_rejection_feedback_reaches_every_planning_step(runs_root: Path, tmp_path: Path) -> None:
    run = seed(runs_root, tmp_path)
    llm = MockLLM()
    llm.add(lambda r: schema(r) == "PlanCore", core)
    llm.add(lambda r: schema(r) == "ResourcePlan", resources)
    llm.add(
        lambda r: schema(r) == "ComputeEstimate",
        {"gpu_memory_gb": 14, "gpus_per_run": 1, "hours_per_run": 1.5, "justification": "BERT base fits"},
    )
    Engine(run, services(run, llm)).advance()
    queue(run, ActionKind.REJECT, payload={"feedback": "Use DistilBERT and keep each run under an hour"})
    first = len(llm.calls)
    assert Engine(run, services(run, llm)).advance().outcome is Outcome.WAITING_GATE
    redo = {schema(r): r.messages[-1].content for r in llm.calls[first:]}
    assert set(redo) == {"PlanCore", "ResourcePlan", "ComputeEstimate"}
    assert all("Use DistilBERT and keep each run under an hour" in text for text in redo.values())
