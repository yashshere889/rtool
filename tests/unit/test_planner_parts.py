"""Planner checks (compute fitting, design rules) and the resource checker / fetcher."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from research_tool.agents.planner.checks import (
    UNCHECKED,
    blocking_issues,
    brief_issues,
    compute_issues,
    design_issues,
    fit_compute,
    format_time,
    metric_issues,
    normalise_core,
    one_primary,
    resource_issues,
    results_schema,
    slug,
)
from research_tool.agents.tasks.fetch_resources import torchvision_downloader
from research_tool.core.config import ExperimentsConfig, load_config
from research_tool.schemas.common import ResearchBrief
from research_tool.schemas.plan import ComputeEstimate, Condition, Metric, PlanCore, ResourceChoice, ResourceSpec
from research_tool.tools.http import PoliteHttp
from research_tool.tools.resources.checks import HubChecker
from research_tool.tools.resources.fetch import Fetcher, local_models

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


def test_a_plan_must_fit_the_budget_as_the_coder_will_charge_it(config) -> None:  # noqa: ANN001
    """Barkla, 2026-11-12: 6 runs at 2 h under a 12 h cap passed the plan check (12.0), but the jobs ask for their
    time limit (2 h x 1.3 = 2.6 h each, 15.6 in all) and the Coder refused to submit them."""
    tight = load_config(BARKLA, overrides={"budgets": {"run": {"max_gpu_hours": 12}}}, env={"USER": "me"})
    estimate = ComputeEstimate(gpu_memory_gb=5, gpus_per_run=1, hours_per_run=2, justification="")
    request, issues = fit_compute(estimate, runs=6, config=tight)
    assert request.gpu_hours_total == 12 and request.time_limit == "02:36:00"
    assert len(issues) == 1 and "15.6" in issues[0] and issues[0].endswith("the budget is 12")
    assert compute_issues(request, tight) == issues  # a hand edit and the page's live check say the same
    shorter = request.model_copy(update={"hours_per_run": 0.25, "time_limit": "00:20:00", "gpu_hours_total": 1.5})
    assert compute_issues(shorter, tight) == []  # 6 x 20 min = 2 GPU-hours
    assert fit_compute(estimate, runs=6, config=config)[1] == []  # 15.6 fits a 50 h budget


def core_with(metrics: list[dict]) -> PlanCore:
    return PlanCore(
        objective="o",
        design="d",
        independent_variables=[{"name": "x", "description": "x"}],
        dependent_variables=[{"name": "y", "description": "y"}],
        conditions=[{"name": "a", "description": "a"}, {"name": "a", "description": "b"}],
        metrics=[{"higher_is_better": True} | m for m in metrics],
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
    twice = normalise_core(core_with([{"name": "Accuracy", "definition": "a"}, {"name": "accuracy", "definition": ""}]))
    assert "metric names must be unique: ['accuracy']" in design_issues(twice, min_seeds=1)
    with pytest.raises(ValidationError, match="higher_is_better"):  # the model may not leave the direction out
        PlanCore.model_validate(core.model_dump() | {"metrics": [{"name": "acc", "definition": "accuracy"}]})


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


def test_metric_direction_is_checked_against_the_name() -> None:
    barkla = [
        Metric(name="task_success_rate", definition="share of episodes solved", higher_is_better=True),
        Metric(name="collision_frequency", definition="collisions per episode", higher_is_better=True),
        Metric(name="Test Loss", definition="cross-entropy", higher_is_better=True),
        Metric(name="accuracy", definition="test accuracy", higher_is_better=False),
        Metric(name="error_reduction", definition="fewer errors than the baseline", higher_is_better=True),
        Metric(name="so_amplitude", definition="slow-oscillation amplitude", higher_is_better=True),
    ]
    issues = metric_issues(barkla)
    assert len(issues) == 3
    assert issues[0].startswith("metric collision_frequency says higher is better, but a higher collision")
    assert "Test Loss" in issues[1] and "accuracy says lower is better" in issues[2]


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
        if request.url.path == "/quote/GSPC/history":
            return httpx.Response(200, text="<!DOCTYPE html><html><body>quotes</body></html>")
        if request.url.path == "/data/ff.csv":
            return httpx.Response(200, text="date,ret\n19260701,0.09\n")
        if request.url.path == "/pypi/pettingzoo/json":
            return httpx.Response(200, json={"info": {"name": "PettingZoo", "license": "MIT"}})
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
    pip = checker.check(
        ResourceChoice(kind="code", name="p", source="pip", identifier="pettingzoo[mpe]>=1.24", purpose="")
    )
    assert pip.exists and pip.license == "MIT"
    typo = checker.check(ResourceChoice(kind="code", name="p", source="pip", identifier="petingzoo", purpose=""))
    assert not typo.exists and "404" in typo.note
    torchvision = checker.check(
        ResourceChoice(kind="dataset", name="c", source="torchvision", identifier="CIFAR10", purpose="")
    )
    assert torchvision.exists
    url = "https://x.test/quote/GSPC/history"
    page = checker.check(ResourceChoice(kind="dataset", name="S&P", source="url", identifier=url, purpose=""))
    assert not page.exists and "HTML" in page.note
    csv = checker.check(
        ResourceChoice(kind="dataset", name="ff", source="url", identifier="https://x.test/data/ff.csv", purpose="")
    )
    assert csv.exists and csv.note == "reachable"
    # nothing public was named: never "verified", whatever it is called
    manual = checker.check(
        ResourceChoice(kind="dataset", name="CS:GO env", source="manual", identifier="custom", purpose="")
    )
    assert not manual.exists and "someone must provide it" in manual.note


def test_a_missing_hub_id_comes_back_with_real_ones() -> None:
    searched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/datasets":
            term = request.url.params["search"]
            searched.append(term)
            return httpx.Response(200, json=[{"id": "armanc/scientific_papers"}] if term == "scientific_papers" else [])
        return httpx.Response(404)

    checker = HubChecker(
        PoliteHttp(
            client=httpx.Client(transport=httpx.MockTransport(handler)), default_interval_s=0, sleep=lambda _s: None
        )
    )
    guess = ResourceChoice(
        kind="dataset", name="scientific_papers", source="huggingface", identifier="nope/tldr", purpose=""
    )
    result = checker.check(guess)
    assert not result.exists and "404" in result.note
    assert "real ids that may fit: armanc/scientific_papers" in result.note
    assert searched == ["scientific_papers"]  # the name found one, so the identifier was not searched too
    other = ResourceChoice(kind="dataset", name="x", source="huggingface", identifier="nope/y", purpose="")
    assert "real ids" not in checker.check(other).note  # nothing similar: just the 404


def test_the_experiments_own_code_is_not_a_manual_resource() -> None:
    code = ResourceSpec(kind="code", name="audit_pipeline", source="manual", identifier="", purpose="")
    data = ResourceSpec(kind="dataset", name="imdb", source="huggingface", identifier="imdb", purpose="", verified=True)
    issues = resource_issues([code, data], max_dataset_gb=100)
    assert len(issues) == 1 and "rtool writes it" in issues[0]


def test_manual_resources_are_an_issue_not_a_verified_resource() -> None:
    manual = ResourceSpec(
        kind="dataset", name="CS:GO env", source="manual", identifier="custom", purpose="", verified=True
    )
    issues = resource_issues([manual], max_dataset_gb=100)
    assert len(issues) == 1 and "marked manual" in issues[0] and "pip package" in issues[0]


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


def test_torchvision_datasets_are_fetched_once_for_every_run(tmp_path: Path) -> None:
    """Barkla, 2026-11-12: CIFAR-10 was "deferred", nothing downloaded it, and the Coder's smoke test found no data
    in four repair rounds. Now the fetch stage downloads it with torchvision and the Coder gets its folder."""

    def downloaded(identifier: str, target: Path) -> str:
        target.mkdir(parents=True)
        (target / "data_batch_1").write_bytes(b"x" * 10)
        return "CIFAR10"

    def unknown(identifier: str, target: Path) -> str:
        raise RuntimeError(f"torchvision download failed: torchvision has no dataset {identifier!r}")

    resource = ResourceSpec(kind="dataset", name="CIFAR-10", source="torchvision", identifier="cifar-10", purpose="")
    ready = Fetcher(tmp_path, snapshot=None, torchvision=downloaded).fetch(resource)
    assert ready.status == "ready" and ready.path == str(tmp_path / "torchvision" / "cifar-10") and ready.files == 1
    assert ready.note == "torchvision.datasets.CIFAR10"
    failed = Fetcher(tmp_path, snapshot=None, torchvision=unknown).fetch(
        resource.model_copy(update={"identifier": "x"})
    )
    assert failed.status == "failed" and "has no dataset 'x'" in failed.note
    assert Fetcher(tmp_path, snapshot=None).fetch(resource).status == "deferred"  # nothing here to download with


def test_the_torchvision_downloader_runs_torchvision_itself(tmp_path: Path) -> None:
    logs = tmp_path / "logs"  # the downloader's working folder: a stand-in torchvision is imported from it
    (logs / "torchvision").mkdir(parents=True)
    (logs / "torchvision" / "__init__.py").write_text("")
    (logs / "torchvision" / "datasets.py").write_text(
        "import os\n"
        "class CIFAR10:\n"
        "    def __init__(self, root, train=True, transform=None, download=False):\n"
        "        open(os.path.join(root, 'train' if train else 'test'), 'w').close()\n"
        "class SVHN:\n"
        "    def __init__(self, root, split='train', download=False):\n"
        "        open(os.path.join(root, split + '.mat'), 'w').close()\n"
        "class ImageNet:\n"
        "    def __init__(self, root, split='train'):\n"
        "        pass\n"
    )
    download = torchvision_downloader(ExperimentsConfig(), tmp_path / "data", logs)
    assert download("cifar-10", tmp_path / "data" / "a") == "CIFAR10"  # matched however the planner spells it
    assert sorted(p.name for p in (tmp_path / "data" / "a").iterdir()) == ["test", "train"]
    assert download("svhn", tmp_path / "data" / "b") == "SVHN"
    assert sorted(p.name for p in (tmp_path / "data" / "b").iterdir()) == ["test.mat", "train.mat"]
    with pytest.raises(RuntimeError, match="cannot download ImageNet itself"):
        download("ImageNet", tmp_path / "data" / "c")
    with pytest.raises(RuntimeError, match="has no dataset 'ImageNet100'"):
        download("ImageNet100", tmp_path / "data" / "d")


def test_brief_limits_name_datasets_loosely_and_check_time_and_seeds() -> None:
    brief = ResearchBrief(topic="t", seeds=5, max_minutes_per_run=10, datasets=["CIFAR-10", "MNIST"])
    ok = [ResourceSpec(kind="dataset", name="cifar10", source="huggingface", identifier="uoft-cs/cifar10", purpose="")]
    other = ResourceSpec(
        kind="dataset", name="ImageNet-100", source="torchvision", identifier="ImageNet100", purpose=""
    )
    model = ResourceSpec(
        kind="model", name="ResNet", source="huggingface", identifier="microsoft/resnet-18", purpose=""
    )
    assert brief_issues(brief, resources=[*ok, model], hours_per_run=10 / 60, n_seeds=5) == []
    issues = brief_issues(brief, resources=[*ok, other], hours_per_run=0.5, n_seeds=3)
    assert len(issues) == 3 and "ImageNet-100" in issues[1] and "30 minutes" in issues[2]
    assert brief_issues(None, n_seeds=1) == brief_issues(ResearchBrief(topic="t"), n_seeds=1, hours_per_run=9) == []


def test_a_model_in_the_llm_cache_is_used_where_it_is(tmp_path: Path) -> None:
    repo = tmp_path / "hf" / "hub" / "models--nvidia--Nemotron-Nano"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("abc")
    (repo / "snapshots" / "abc").mkdir(parents=True)
    (repo / "snapshots" / "abc" / "model.safetensors").write_bytes(b"x" * 10)
    local = local_models(str(tmp_path / "hf"))
    assert local == {"nvidia/Nemotron-Nano": str(repo / "snapshots" / "abc")}
    assert local_models(None) == local_models(str(tmp_path / "nowhere")) == {}

    def no_download(**_: object) -> str:
        raise AssertionError("downloaded a model the cluster already has")

    spec = ResourceSpec(
        kind="model", name="Nemotron", source="huggingface", identifier="nvidia/Nemotron-Nano", purpose=""
    )
    got = Fetcher(tmp_path / "res", snapshot=no_download, local=local).fetch(spec)
    assert (got.status, got.path, got.bytes, got.note) == (
        "ready",
        local["nvidia/Nemotron-Nano"],
        10,
        "already on the cluster",
    )


def test_ablations_that_repeat_a_condition_are_dropped() -> None:  # seen on Barkla: 7 of 31 validity runs
    core = core_with([{"name": "acc", "definition": "a"}])
    base, new = Condition(name="Base", description="b", is_baseline=True), Condition(name="new", description="n")
    core.conditions = [base, new]
    core.ablations = [Condition(name="base", description="b"), Condition(name="no_x", description="x")]
    assert [a.name for a in normalise_core(core).ablations] == ["no_x"]


def test_only_a_data_analysis_must_list_a_dataset() -> None:  # a simulation makes its own data
    numpy = ResourceSpec(kind="code", name="numpy", source="pip", identifier="numpy", purpose="", verified=True)
    assert resource_issues([numpy], max_dataset_gb=100) == []
    assert "no dataset is listed" in resource_issues([numpy], max_dataset_gb=100, units=True)[0]


def test_the_brief_can_limit_models_and_forbid_downloads() -> None:
    nemo = ResourceSpec(
        kind="model", name="Nemotron", source="huggingface", identifier="nvidia/NVIDIA-Nemotron-Nano-12B-v2", purpose=""
    )
    bert = ResourceSpec(kind="model", name="BERT", source="huggingface", identifier="bert-base-uncased", purpose="")
    gsm = ResourceSpec(kind="dataset", name="GSM8K", source="huggingface", identifier="openai/gsm8k", purpose="")
    pkg = ResourceSpec(kind="code", name="vllm", source="pip", identifier="vllm", purpose="")

    only = ResearchBrief(topic="t", models=["nvidia/NVIDIA-Nemotron-Nano-12B-v2"])
    assert brief_issues(only, resources=[nemo, pkg]) == []
    assert "BERT is not one of the models" in brief_issues(only, resources=[nemo, bert])[0]
    assert "Use only these models" in only.requirements()[0]

    nothing = ResearchBrief(topic="t", no_download=True)
    assert brief_issues(nothing, resources=[pkg]) == []
    assert len(brief_issues(nothing, resources=[gsm, nemo, pkg])) == 2
    assert "List no dataset or model" in nothing.requirements()[0]


def test_blocking_issues_are_what_the_researcher_must_acknowledge() -> None:
    missing = ResourceSpec(kind="dataset", name="C-Eval", source="huggingface", identifier="hfl/ceval", purpose="")
    missing.note = "HTTP 404"
    manual = ResourceSpec(kind="dataset", name="CS:GO env", source="manual", identifier="x", purpose="")
    own_code = ResourceSpec(kind="code", name="audit", source="manual", identifier="", purpose="")
    gated = ResourceSpec(kind="model", name="Llama", source="huggingface", identifier="meta-llama/L", purpose="")
    gated.verified, gated.note = True, "gated (manual): needs HF_TOKEN and accepted terms"
    by_hand = ResourceSpec(kind="dataset", name="PubMed", source="url", identifier="https://x/y.parquet", purpose="")
    by_hand.note = UNCHECKED
    fine = ResourceSpec(kind="dataset", name="MNIST", source="torchvision", identifier="MNIST", purpose="")
    fine.verified = True
    found = blocking_issues([missing, manual, own_code, gated, by_hand, fine])
    assert len(found) == 3 and "HTTP 404" in found[0] and "not public" in found[1] and "gated" in found[2]


def test_a_hypothesis_that_names_data_the_brief_forbids_is_flagged() -> None:
    from research_tool.agents.hypothesis.agent import _HypothesisRun
    from research_tool.schemas.hypothesis import Hypothesis

    def hypothesis(*datasets: str) -> Hypothesis:
        return Hypothesis.model_construct(id="h1", datasets=list(datasets))

    run = object.__new__(_HypothesisRun)
    run.brief = ResearchBrief(topic="t", datasets=["MNIST", "CIFAR-10"])
    assert run.brief_conflicts(hypothesis("CIFAR-10")) == []
    assert "ImageNet-100" in run.brief_conflicts(hypothesis("CIFAR-10", "ImageNet-100"))[0]
    run.brief = ResearchBrief(topic="t", no_download=True)
    assert run.brief_conflicts(hypothesis()) == [] and "GSM8K" in run.brief_conflicts(hypothesis("GSM8K"))[0]


def test_models_must_be_hub_ids_and_gated_ones_are_an_issue_unless_cached() -> None:
    local = ResourceSpec(kind="model", name="8B model", source="local", identifier="8B-parameter model", purpose="")
    described = ResourceSpec(
        kind="model", name="4B model", source="huggingface", identifier="4B-parameter BF16 model", purpose=""
    )
    gated = ResourceSpec(kind="model", name="Llama", source="huggingface", identifier="meta-llama/L", purpose="")
    gated.verified, gated.note = True, "gated (manual): needs HF_TOKEN and accepted terms"
    issues = resource_issues([local, described, gated], max_dataset_gb=100)
    assert len(issues) == 3 and "Hugging Face id" in issues[0] and "description" in issues[1] and "gated" in issues[2]
    assert resource_issues([gated], max_dataset_gb=100, cached={"meta-llama/L"}) == []


def test_the_hub_search_returns_ids_with_tags() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/models" and request.url.params["search"] == "Nemotron"
        return httpx.Response(200, json=[{"id": "nvidia/N", "downloads": 5, "pipeline_tag": "text-generation"}, {}])

    checker = HubChecker(
        PoliteHttp(
            client=httpx.Client(transport=httpx.MockTransport(handler)), default_interval_s=0, sleep=lambda _s: None
        )
    )
    assert checker.search("model", "Nemotron") == [
        {"id": "nvidia/N", "downloads": 5, "pipeline_tag": "text-generation", "gated": None}
    ]


def test_a_torchvision_name_that_never_existed_is_not_found_at_plan_time() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    checker = HubChecker(
        PoliteHttp(
            client=httpx.Client(transport=httpx.MockTransport(handler)), default_interval_s=0, sleep=lambda _s: None
        )
    )

    def check(name: str):  # noqa: ANN202
        return checker.check(
            ResourceChoice(kind="dataset", name=name, source="torchvision", identifier=name, purpose="")
        )

    assert check("ImageNet100").exists is False and "no dataset 'ImageNet100'" in check("ImageNet100").note
    assert check("CIFAR-10").exists and check("cifar10").exists and check("MNIST").exists


def test_torchvision_list_matches_the_installed_torchvision() -> None:
    datasets = pytest.importorskip("torchvision.datasets")
    from research_tool.tools.resources.checks import TORCHVISION_DATASETS

    downloadable = {
        n
        for n in dir(datasets)
        if isinstance(getattr(datasets, n), type) and "download" in getattr(datasets, n).__init__.__code__.co_varnames
    }
    assert downloadable <= set(TORCHVISION_DATASETS)


def test_a_dataset_that_is_only_a_loading_script_is_an_issue() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        files = {
            "/api/datasets/old/scripted": ["scripted.py", "README.md"],
            "/api/datasets/new/parquet": ["data/train.parquet"],
        }
        if request.url.path in files:
            return httpx.Response(
                200, json={"siblings": [{"rfilename": f, "size": 10} for f in files[request.url.path]]}
            )
        return httpx.Response(404)

    checker = HubChecker(
        PoliteHttp(
            client=httpx.Client(transport=httpx.MockTransport(handler)), default_interval_s=0, sleep=lambda _s: None
        )
    )
    old = checker.check(
        ResourceChoice(kind="dataset", name="o", source="huggingface", identifier="old/scripted", purpose="")
    )
    new = checker.check(
        ResourceChoice(kind="dataset", name="n", source="huggingface", identifier="new/parquet", purpose="")
    )
    assert old.note.startswith("script-only") and new.note == ""
    spec = ResourceSpec(kind="dataset", name="o", source="huggingface", identifier="old/scripted", purpose="")
    spec.verified, spec.note = True, old.note
    assert "loading script" in resource_issues([spec], 100)[0]


def test_the_experiments_own_output_is_not_a_local_dataset() -> None:
    out = ResourceSpec(kind="dataset", name="Simulation Results", source="local", identifier="results.json", purpose="")
    out2 = ResourceSpec(
        kind="dataset", name="Queue", source="local", identifier="queue_simulation_results.csv", purpose=""
    )
    real = ResourceSpec(kind="dataset", name="cells", source="local", identifier="cells", purpose="")
    real.verified = True
    issues = resource_issues([out, out2, real], max_dataset_gb=100)
    assert len(issues) == 2 and all("own output" in i for i in issues)
