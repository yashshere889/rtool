"""Planner + fetch_resources: repair round, compute fitting, plan gate, then downloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_tool.agents.planner.checks import UNCHECKED
from research_tool.core.actions import ActionInbox, ActionKind
from research_tool.core.artifacts import ArtifactStore, Producer
from research_tool.core.engine import Engine, Outcome, load_status
from research_tool.core.events import EventLog
from research_tool.core.run import RunHandle
from research_tool.core.services import Services
from research_tool.knowledge.store import KnowledgeStore
from research_tool.llm.client import LLMClient
from research_tool.llm.mock_backend import MockLLM
from research_tool.llm.types import ChatRequest
from research_tool.schemas.common import SelectedItem
from research_tool.schemas.hypothesis import CheckedEvidence, Hypothesis
from research_tool.schemas.literature import Corpus, Entity, KnowledgeMap, PaperRecord
from research_tool.schemas.plan import ExperimentPlan
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
    store = ArtifactStore(run.artifacts_dir)
    noted = store.get("selected_hypothesis").model_copy(update={"note": "Prefer public datasets"})
    store.put("selected_hypothesis", noted, producer=Producer(stage="hypotheses", agent="test"))
    Engine(run, services(run, llm)).advance()
    assert llm.calls and all("Prefer public datasets" in r.messages[-1].content for r in llm.calls)  # the choice's note
    queue(run, ActionKind.REJECT, payload={"feedback": "Use DistilBERT and keep each run under an hour"})
    first = len(llm.calls)
    assert Engine(run, services(run, llm)).advance().outcome is Outcome.WAITING_GATE
    redo = {schema(r): r.messages[-1].content for r in llm.calls[first:]}
    assert set(redo) == {"PlanCore", "ResourcePlan", "ComputeEstimate"}
    assert all("Use DistilBERT and keep each run under an hour" in text for text in redo.values())


def planned(runs_root: Path, tmp_path: Path) -> tuple[RunHandle, MockLLM]:
    run = seed(runs_root, tmp_path)
    llm = MockLLM()
    llm.add(lambda r: schema(r) == "PlanCore", core)
    llm.add(lambda r: schema(r) == "ResourcePlan", resources)
    llm.add(
        lambda r: schema(r) == "ComputeEstimate",
        {"gpu_memory_gb": 14, "gpus_per_run": 1, "hours_per_run": 1.5, "justification": "BERT base fits"},
    )
    assert Engine(run, services(run, llm)).advance().outcome is Outcome.WAITING_GATE
    return run, llm


def test_a_hand_edited_plan_is_normalised_and_checked(runs_root: Path, tmp_path: Path) -> None:
    run, llm = planned(runs_root, tmp_path)
    store = ArtifactStore(run.artifacts_dir)
    original = store.get("experiment_plan")
    value = original.model_dump(mode="json")
    value["conditions"].append({"name": "Low Noise", "description": "5% noise", "is_baseline": False})
    value["seeds"] = [0, 1]  # fewer than the planner asks for: a warning, not a refusal
    value["resources"][0]["identifier"] = "stanfordnlp/imdb-plain"  # nobody has checked this one
    value["compute"]["hours_per_run"] = 3.0
    value["hypothesis"] = "something else"  # comes from the chosen hypothesis, not editable here
    value["results_schema"] = {}  # derived
    edit = {"artifact": "experiment_plan", "base_version": 1, "value": value, "note": "add a milder condition"}
    queue(run, ActionKind.EDIT, payload=edit)
    queue(run, ActionKind.EDIT, payload=edit)  # made on version 1, which is no longer the latest
    queue(run, ActionKind.EDIT, payload={**edit, "artifact": "corpus"})
    calls = len(llm.calls)
    assert Engine(run, services(run, llm)).advance().outcome is Outcome.WAITING_GATE
    assert len(llm.calls) == calls  # an edit needs no model

    outcomes = [a["outcome"] for a in ActionInbox(run.actions_dir).processed()]
    assert outcomes[0]["status"] == "applied" and outcomes[0]["version"] == 2
    assert "changed since you started editing" in outcomes[1]["error"]
    assert "cannot be edited by hand" in outcomes[2]["error"]

    envelope = store.envelope("experiment_plan")
    plan = ExperimentPlan.model_validate(envelope.payload)
    assert envelope.producer.agent.startswith("human:") and envelope.producer.stage == "plan"
    assert envelope.meta["edited_from"] == 1 and envelope.meta["edit_note"] == "add a milder condition"
    assert envelope.inputs == store.envelope("experiment_plan", 1).inputs
    assert [c.name for c in plan.conditions] == ["clean_labels", "noisy_labels", "low_noise"]
    assert plan.hypothesis == original.hypothesis
    assert (plan.compute.runs, plan.compute.gpu_hours_total, plan.compute.time_limit) == (6, 18.0, "03:54:00")
    assert plan.compute.partition == "gpu-l40s"  # not picked again
    assert plan.results_schema["properties"]["runs"]["minItems"] == 6
    imdb = plan.resources[0]
    assert not imdb.verified and imdb.note == UNCHECKED and plan.resources[1].verified
    assert any("at least 3 seeds" in i for i in plan.issues) and any("changed by hand" in i for i in plan.issues)
    assert "low_noise" in (run.root / plan.markdown_path).read_text()

    _, stages, _ = load_status(run)
    gate = stages["plan"].gate
    assert gate is not None and gate.summary["artifacts"] == {"experiment_plan": 2}  # the checkpoint shows the edit
    queue(run, ActionKind.APPROVE, payload={"expect": {"artifacts": {"experiment_plan": 1}}})  # seen before the edit
    queue(run, ActionKind.APPROVE, payload={"expect": {"artifacts": {"experiment_plan": 2}}})
    assert Engine(run, services(run, llm)).advance().outcome is Outcome.DONE
    assert store.envelope("resource_manifest").inputs == {"experiment_plan": 2}


def test_editing_a_finished_plan_makes_later_stages_run_again(runs_root: Path, tmp_path: Path) -> None:
    run, llm = planned(runs_root, tmp_path)
    queue(run, ActionKind.APPROVE)
    assert Engine(run, services(run, llm)).advance().outcome is Outcome.DONE
    store = ArtifactStore(run.artifacts_dir)
    value = store.get("experiment_plan").model_dump(mode="json")
    value["expected_results"] = "noisy is clearly worse"
    queue(run, ActionKind.EDIT, payload={"artifact": "experiment_plan", "base_version": 1, "value": value})
    assert Engine(run, services(run, llm)).advance().outcome is Outcome.DONE
    edited = [e for e in EventLog(run.events_path, run.run_id) if e["kind"] == "artifact_edited"]
    _, stages, _ = load_status(run)
    assert edited[0]["data"]["stale"] == ["fetch"] and stages["fetch"].input_versions == {"experiment_plan": 2}

    same = {"artifact": "experiment_plan", "base_version": 2, "value": store.get("experiment_plan").model_dump()}
    queue(run, ActionKind.EDIT, payload=same)
    Engine(run, services(run, llm)).advance()
    assert ActionInbox(run.actions_dir).processed()[-1]["outcome"]["unchanged"] and store.head("experiment_plan") == 2


def with_manual_env(request: ChatRequest) -> dict[str, Any]:
    plan = resources(request)
    plan["resources"].append(
        {"kind": "other", "name": "CS:GO env", "source": "manual", "identifier": "custom", "purpose": "simulator"}
    )
    return plan


def manual_plan_llm() -> MockLLM:
    llm = MockLLM()
    llm.add(lambda r: schema(r) == "PlanCore", core)
    llm.add(lambda r: schema(r) == "ResourcePlan", with_manual_env)
    llm.add(
        lambda r: schema(r) == "ComputeEstimate",
        {"gpu_memory_gb": 14, "gpus_per_run": 1, "hours_per_run": 1.5, "justification": "BERT base fits"},
    )
    return llm


def test_manual_resources_stop_the_fetch_until_someone_decides(runs_root: Path, tmp_path: Path) -> None:
    run = seed(runs_root, tmp_path)
    llm = manual_plan_llm()
    assert Engine(run, services(run, llm)).advance().open_gates[0]["gate"] == "plan_gate"
    plan = ArtifactStore(run.artifacts_dir).get("experiment_plan")
    env = next(r for r in plan.resources if r.source == "manual")
    assert any("CS:GO env is marked manual" in i for i in plan.issues)  # flagged at the plan gate already
    assert "NOT PUBLIC" in (run.root / plan.markdown_path).read_text()
    assert env.name == "CS:GO env"

    queue(run, ActionKind.APPROVE)
    stopped = Engine(run, services(run, llm)).advance()
    gate = stopped.open_gates[0]
    assert stopped.outcome is Outcome.WAITING_GATE and gate["gate"] == "manual_resources" and gate["in_stage"]
    folder = Path(gate["summary"]["missing"][0]["path"])
    assert folder.is_dir() and not any(folder.iterdir()) and str(folder) in gate["summary"]["message"]

    (folder / "simulator.py").write_text("print('the researcher brought it')\n")
    queue(run, ActionKind.APPROVE)
    assert Engine(run, services(run, llm)).advance().outcome is Outcome.DONE
    manifest = ArtifactStore(run.artifacts_dir).get("resource_manifest")
    provided = next(r for r in manifest.resources if r.name == "CS:GO env")
    assert provided.status == "manual" and provided.files == 1 and provided.note == "provided by hand (1 files)"


def test_rejecting_the_manual_resources_sends_the_plan_back(runs_root: Path, tmp_path: Path) -> None:
    run = seed(runs_root, tmp_path)
    llm = manual_plan_llm()
    Engine(run, services(run, llm)).advance()
    queue(run, ActionKind.APPROVE)
    Engine(run, services(run, llm)).advance()  # stops at manual_resources
    queue(run, ActionKind.REJECT, payload={"feedback": "use PettingZoo MPE (pip install pettingzoo)"})
    again = Engine(run, services(run, llm)).advance()
    assert again.open_gates[0]["gate"] == "plan_gate"  # the plan was redone and waits for approval again
    asked = [r.messages[-1].content for r in llm.calls if schema(r) == "PlanCore"]
    assert "use PettingZoo MPE (pip install pettingzoo)" in asked[-1]
