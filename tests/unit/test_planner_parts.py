"""Planner checks (compute fitting, design rules) and the resource checker / fetcher."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from research_tool.agents.planner.checks import (
    design_issues,
    fit_compute,
    format_time,
    normalise_core,
    one_primary,
    results_schema,
    slug,
)
from research_tool.core.config import load_config
from research_tool.schemas.plan import ComputeEstimate, Condition, Metric, PlanCore, ResourceChoice, ResourceSpec
from research_tool.tools.http import PoliteHttp
from research_tool.tools.resources.checks import HubChecker
from research_tool.tools.resources.fetch import Fetcher

BARKLA = Path(__file__).resolve().parents[2] / "configs" / "barkla.yaml"


@pytest.fixture
def config():
    return load_config(BARKLA, overrides={"budgets": {"run": {"max_gpu_hours": 50}}}, env={"USER": "me"})


def test_format_time_and_slug() -> None:
    assert format_time(9360) == "02:36:00" and format_time(3 * 86400) == "3-00:00:00" and format_time(61) == "00:02:00"
    assert slug("With Stimulation!") == "with_stimulation"


@pytest.mark.parametrize(
    ("memory", "gpus", "partition"),
    [(30, 1, "gpu-l40s"), (60, 1, "gpu-h100"), (12, 1, "gpu-l40s"), (40, 4, "gpu-h100"), (0, 0, "nodes")],
)
def test_first_preferred_partition_that_fits(config, memory: float, gpus: int, partition: str) -> None:  # noqa: ANN001
    request, issues = fit_compute(
        ComputeEstimate(gpu_memory_gb=memory, gpus_per_run=gpus, hours_per_run=1, justification="x"),
        runs=6,
        config=config,
    )
    assert request.partition == partition and not issues


def test_without_a_cluster_profile_nothing_is_flagged() -> None:
    request, issues = fit_compute(
        ComputeEstimate(gpu_memory_gb=8, gpus_per_run=1, hours_per_run=1, justification=""),
        runs=2,
        config=load_config(env={}),
    )
    assert request.partition == "gpu-l40s" and not issues


def test_compute_problems_are_reported(config) -> None:  # noqa: ANN001
    request, issues = fit_compute(
        ComputeEstimate(gpu_memory_gb=2, gpus_per_run=1, hours_per_run=2, justification=""), runs=9, config=config
    )
    assert request.time_limit == "02:36:00" and request.gpu_hours_total == 18 and not issues
    too_big, issues = fit_compute(
        ComputeEstimate(gpu_memory_gb=200, gpus_per_run=1, hours_per_run=1, justification=""), runs=1, config=config
    )
    assert too_big.partition == "gpu-h100" and "no allowed partition" in issues[0]
    _, issues = fit_compute(
        ComputeEstimate(gpu_memory_gb=10, gpus_per_run=1, hours_per_run=60, justification=""), runs=1, config=config
    )
    assert any("allows 3-00:00:00" in i for i in issues) and any("budget is 50" in i for i in issues)


def core_with(metrics: list[dict]) -> PlanCore:
    return PlanCore(
        objective="o",
        design="d",
        independent_variables=[{"name": "x", "description": "x"}],
        dependent_variables=[{"name": "y", "description": "y"}],
        conditions=[{"name": "a", "description": "a"}, {"name": "a", "description": "b"}],
        metrics=metrics,
        statistical_tests=[{"test": "t", "comparison": "a vs b"}],
        n_seeds=1,
        procedure=["1", "2", "3"],
        expected_results="r",
    )


def test_design_rules() -> None:
    core = core_with([{"name": "acc", "definition": "accuracy"}])
    issues = " ".join(design_issues(core, min_seeds=3))
    assert "baseline" in issues and "unique" in issues and "at least 3 seeds" in issues
    assert "primary" not in issues


def test_exactly_one_primary_metric() -> None:
    unmarked = normalise_core(core_with([{"name": "Acc", "definition": "a"}, {"name": "F1", "definition": "f"}]))
    assert [m.primary for m in unmarked.metrics] == [True, False]  # none marked: the first one is primary
    marked = normalise_core(
        core_with([{"name": "acc", "definition": "a"}, {"name": "f1", "definition": "f", "primary": True}])
    )
    assert [m.primary for m in marked.metrics] == [False, True]  # the planner's choice is kept
    several = normalise_core(
        core_with(
            [{"name": "acc", "definition": "a", "primary": True}, {"name": "f1", "definition": "f", "primary": True}]
        )
    )
    assert any("exactly one metric as the primary outcome" in i for i in design_issues(several, min_seeds=1))
    assert [m.primary for m in one_primary(several.metrics)] == [True, False]  # unresolved after repair: the first


def test_results_schema_shape() -> None:
    schema = results_schema(
        [Condition(name="base", description="", is_baseline=True), Condition(name="t", description="")],
        [Metric(name="acc", definition="")],
        [0, 1, 2],
    )
    runs = schema["properties"]["runs"]
    assert runs["minItems"] == 6 and runs["items"]["properties"]["condition"]["enum"] == ["base", "t"]
    assert runs["items"]["properties"]["metrics"]["required"] == ["acc"]


def test_hub_checker() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/datasets/imdb":
            return httpx.Response(200, json={"cardData": {"license": "other"}, "usedStorage": 2_500_000_000})
        if request.url.path == "/api/models/meta-llama/x":
            return httpx.Response(
                200, json={"gated": "manual", "tags": ["license:llama3"], "siblings": [{"size": 10**9}]}
            )
        return httpx.Response(404)

    checker = HubChecker(
        PoliteHttp(
            client=httpx.Client(transport=httpx.MockTransport(handler)), default_interval_s=0, sleep=lambda _s: None
        )
    )
    imdb = checker.check(
        ResourceChoice(kind="dataset", name="imdb", source="huggingface", identifier="imdb", purpose="")
    )
    assert imdb.exists and imdb.license == "other" and imdb.size_gb == 2.5
    gated = checker.check(
        ResourceChoice(kind="model", name="m", source="huggingface", identifier="meta-llama/x", purpose="")
    )
    assert gated.exists and gated.license == "llama3" and gated.size_gb == 1.0 and "gated" in gated.note
    missing = checker.check(
        ResourceChoice(kind="dataset", name="n", source="huggingface", identifier="nope/x", purpose="")
    )
    assert not missing.exists and "404" in missing.note
    pip = checker.check(ResourceChoice(kind="code", name="p", source="pip", identifier="scikit-learn", purpose=""))
    assert pip.exists


def test_fetcher(tmp_path: Path) -> None:
    def fake_snapshot(repo_id: str, repo_type: str, local_dir: str, token: str | None) -> str:
        Path(local_dir).mkdir(parents=True)
        (Path(local_dir) / "train.parquet").write_bytes(b"x" * 100)
        return local_dir

    http = PoliteHttp(
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"abc"))),
        default_interval_s=0,
        sleep=lambda _s: None,
    )
    fetcher = Fetcher(tmp_path, http=http, snapshot=fake_snapshot)

    def spec(**kw: str) -> ResourceSpec:
        return ResourceSpec(kind=kw.pop("kind", "dataset"), name=kw.pop("name", "r"), purpose="", **kw)  # type: ignore[arg-type]

    hf = fetcher.fetch(spec(source="huggingface", identifier="org/data"))
    assert hf.status == "ready" and hf.bytes == 100 and hf.files == 1 and "datasets/org--data" in hf.path
    assert fetcher.fetch(spec(source="pip", identifier="numpy")).status == "deferred"
    manual = fetcher.fetch(spec(source="manual", identifier="?", name="private set"))
    assert manual.status == "manual" and Path(manual.path).is_dir()
    url = fetcher.fetch(spec(source="url", identifier="https://data.example.org/files/d.csv", name="csv"))
    assert url.status == "ready" and Path(url.path).read_bytes() == b"abc"
    no_hub = Fetcher(tmp_path, snapshot=None)
    no_hub.snapshot = None
    assert "huggingface_hub is not installed" in no_hub.fetch(spec(source="huggingface", identifier="a/b")).note
