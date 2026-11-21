"""A protein case through the real pipeline, in the pack's own image, with a UniProt sequence fetched.

The demo model answers, except for the design and the experiment (both written here); UniProt is a mock
server in its documented format. The experiment environment records which image the Coder chose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from research_tool.agents.coder.env import ExperimentEnv
from research_tool.core.artifacts import ArtifactStore
from research_tool.core.config import deep_merge, load_config
from research_tool.core.engine import Outcome
from research_tool.core.run import RunHandle
from research_tool.testing.demo_run import CLUSTER, advance, demo_services
from research_tool.testing.samples import structures_mock
from research_tool.tools.resources.fetch import Fetcher
from research_tool.validity.cases import Case, load_case
from research_tool.validity.runs import create_case_run
from research_tool.validity.score import score_run

CASE = Path(__file__).resolve().parents[2] / "benchmarks" / "protein" / "motif-classifier"
IMAGE = {"container": "/images/protein.sif", "packages": ["torch", "esm", "biopython"]}
DESIGN = {
    "objective": "Compare dipeptide features with length for detecting the motif",
    "design": "Two classifiers on the same split; the seed draws the training subset",
    "independent_variables": [
        {"name": "features", "description": "length or dipeptides", "values": ["length", "dipeptide"]}
    ],
    "dependent_variables": [{"name": "test_accuracy", "description": "accuracy on test.csv"}],
    "controls": ["the same split"],
    "conditions": [
        {"name": "length_only", "description": "length", "is_baseline": True},
        {"name": "dipeptide", "description": "dipeptide counts"},
    ],
    "ablations": [],
    "metrics": [{"name": "test_accuracy", "definition": "accuracy", "higher_is_better": True, "primary": True}],
    "statistical_tests": [{"test": "paired t-test", "comparison": "dipeptide vs length_only"}],
    "n_seeds": 8,
    "procedure": ["Read the data", "Train on a seeded subset", "Score test.csv", "Count test sequences in train"],
    "expected_results": "Dipeptides beat length",
    "risks": ["synthetic sequences"],
}
RESOURCES = {
    "resources": [
        {
            "kind": "dataset",
            "name": "motif_sequences",
            "source": "local",
            "identifier": "motif_sequences",
            "purpose": "data",
        },
        # not a dataset: the brief allows only motif_sequences as data (a dataset here is rightly a plan issue)
        {
            "kind": "other",
            "name": "hemoglobin",
            "source": "uniprot",
            "identifier": "P69905",
            "purpose": "a real sequence",
        },
    ]
}
COMPUTE = {"gpu_memory_gb": 0, "gpus_per_run": 0, "hours_per_run": 0.05, "justification": "tiny classifiers"}
EXPERIMENT = '''"""Motif detection by dipeptides or by length (written for this test)."""
# requirements:
import csv
import random
from pathlib import Path

LEAK = {leak}  # test sequences copied into training: 0 is a clean split


def rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def run(condition, seed, resources, smoke, workdir):
    folder = Path(resources["motif_sequences"])
    train, test = rows(folder / "train.csv"), rows(folder / "test.csv")
    assert (Path(resources["hemoglobin"]).read_text().startswith(">sp|P69905"))  # the fetched UniProt record
    rng = random.Random(seed)
    train = rng.sample(train, len(train) * 3 // 4) + test[:LEAK]
    if condition == "length_only":  # the seed moves the cut, as it would move a fitted threshold
        cut = sorted(len(r["sequence"]) for r in train)[int(len(train) * rng.uniform(0.3, 0.7))]
        predict = lambda s: len(s) > cut  # noqa: E731
    else:
        score = lambda s: sum(s.count(p) for p in ("HE", "EG", "GH"))  # noqa: E731
        best = max(range(6), key=lambda c: sum((score(r["sequence"]) > c) == (r["label"] == "1") for r in train))
        predict = lambda s: score(s) > best  # noqa: E731
    accuracy = sum(predict(r["sequence"]) == (r["label"] == "1") for r in test) / len(test)
    seen = {{r["sequence"] for r in train}}
    return {{"test_accuracy": accuracy, "test_sequences_in_train": float(sum(r["sequence"] in seen for r in test))}}
'''


def schema_is(title: str) -> Any:
    return lambda request: (request.json_schema or {}).get("title") == title


def run_case(root: Path, leak: int) -> tuple[RunHandle, list[str]]:
    config = load_config(
        overrides=deep_merge(
            CLUSTER, {"paths": {"runs_root": str(root)}, "experiments": {"pack_images": {"protein": IMAGE}}}
        ),
        env={},
    )
    # the demo experiment's effect is larger than the shipped case's band (which is set for logistic regression)
    data = load_case(CASE).model_dump(mode="json")
    data["expect"]["difference"] = None
    case = Case.model_validate(data)
    case._folder = CASE.resolve()
    run = create_case_run(config, case, commit="0" * 12)
    services = demo_services(run)
    images: list[str] = []

    def environment(cfg: Any, code: Path) -> ExperimentEnv:
        images.append(cfg.container)  # which image the Coder picked; run here without one
        return ExperimentEnv(cfg.model_copy(update={"container": None}), code)

    services.tools["experiment_env"] = environment
    mock = structures_mock()
    services.tools["resource_fetcher"] = lambda target: Fetcher(
        target, http=mock, provided=run.config.experiments.local_data
    )
    for rule in (
        (schema_is("PlanCore"), DESIGN),
        (schema_is("ResourcePlan"), RESOURCES),
        (schema_is("ComputeEstimate"), COMPUTE),
        ("Write the complete Python file `experiment.py`", f"```python\n{EXPERIMENT.format(leak=leak)}```\n"),
    ):
        services.llm.backend.rules.insert(0, rule)
    result = advance(run, services, max_episodes=60)
    assert result.outcome is Outcome.DONE, f"{result.outcome.value}: {result.message}"
    run.prompts = [r.messages[-1].content for r in services.llm.backend.calls]  # type: ignore[attr-defined]
    return run, images


@pytest.fixture(scope="module")
def clean(tmp_path_factory: pytest.TempPathFactory) -> tuple[RunHandle, list[str]]:
    return run_case(tmp_path_factory.mktemp("protein") / "runs", leak=0)


def test_the_coder_works_in_the_packs_own_image(clean: tuple[RunHandle, list[str]]) -> None:
    run, images = clean
    assert images and set(images) == {"/images/protein.sif"}
    coder = next(p for p in run.prompts if "Write the complete Python file" in p)  # type: ignore[attr-defined]
    assert "Installed: torch, esm, biopython" in coder and "test_sequences_in_train" in coder
    bundle = ArtifactStore(run.artifacts_dir).get("code_bundle")
    assert bundle.container == "/images/protein.sif" and bundle.pack == "protein"
    fetched = {r.name: r for r in ArtifactStore(run.artifacts_dir).get("resource_manifest").resources}
    assert fetched["hemoglobin"].status == "ready" and fetched["hemoglobin"].path.endswith("uniprot/P69905.fasta")  # type: ignore[union-attr]
    assert (score_run(run).level, score_run(run).category) == (3, "")


def test_a_split_that_leaks_test_sequences_is_caught(tmp_path: Path) -> None:
    run, _ = run_case(tmp_path / "runs", leak=3)
    results = ArtifactStore(run.artifacts_dir).get("experiment_results")
    assert results.issues and results.issues[0].startswith("`test_sequences_in_train` is 3")
    assert (score_run(run).level, score_run(run).category) == (1, "design_problem")
