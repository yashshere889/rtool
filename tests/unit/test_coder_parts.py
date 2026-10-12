"""The Coder's deterministic parts: statistics, playbook, results, figures, environment, harness."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path

import pytest

from research_tool.agents.coder import stats
from research_tool.agents.coder.agent import extract_code, file_listing, static_problems
from research_tool.agents.coder.env import ExperimentEnv
from research_tool.agents.coder.playbook import (
    declared_requirements,
    diagnose,
    halve_batch_size,
    merge_requirements,
    valid_requirement,
)
from research_tool.agents.coder.results import (
    aggregate,
    analysis_notes,
    correction_used,
    render_figures,
    render_tables,
    results_payload,
    translated_tests,
    validate,
)
from research_tool.core.config import ExperimentsConfig
from research_tool.hpc.sbatch import render_script
from research_tool.packs import available_packs, load_pack
from research_tool.schemas.code import RunRecord
from research_tool.testing.samples import sample_plan as plan
from research_tool.testing.samples import sample_records as records
from research_tool.tools.report.numbers import fmt, fmt_p
from research_tool.tools.report.svg import bar_chart, nice_ticks


# ------------------------------------------------------------------ stats
@pytest.mark.parametrize(("df", "value"), [(1, 12.706), (2, 4.303), (10, 2.228), (30, 2.042)])
def test_t_critical_matches_tables(df: int, value: float) -> None:
    assert stats.t_critical(df) == pytest.approx(value, abs=1e-3)


def test_tests_on_a_known_example() -> None:
    a, b = [0.71, 0.74, 0.69], [0.65, 0.66, 0.63]
    paired = stats.paired_t(a, b)
    assert paired.statistic == pytest.approx(10.0) and paired.p_value == pytest.approx(0.00985, abs=1e-4)
    assert paired.ci_low < paired.difference < paired.ci_high
    assert stats.welch_t(a, b).p_value == pytest.approx(0.0248, abs=1e-3)
    assert stats.wilcoxon_signed_rank(a, b).p_value == pytest.approx(0.25)  # smallest possible with 3 pairs
    assert stats.mann_whitney(a, b).p_value == pytest.approx(0.1)  # complete separation, 3 vs 3
    same = stats.paired_t([1.0, 1.0], [1.0, 1.0])
    assert same.p_value is None and same.difference == 0  # no noise: nothing to test against
    assert stats.holm([0.01, 0.04, 0.03, None]) == pytest.approx([0.03, 0.06, 0.06, None])


def test_no_variation_across_seeds_gives_no_p_value() -> None:
    """Seen on Barkla: 0.6 three times has a computed sd of 7e-17, which gave p = 1e-32 and d = -5.9e15."""
    flat, base = [1.0 - (6 - 2) * 0.1] * 3, [1.0] * 3
    assert stats.stdev(flat) == 0 and stats.mean_ci(flat) == (pytest.approx(0.6), pytest.approx(0.6))
    for result in (stats.paired_t(flat, base), stats.welch_t(flat, base)):
        assert result.p_value is None and result.effect_size is None and result.statistic is None
        assert result.difference == pytest.approx(-0.4)
    tiny = stats.paired_t([1e-9, 2e-9, 3e-9], [0.0, 0.0, 0.0])  # small values still have real spread
    assert tiny.p_value is not None and tiny.effect_size is not None


def test_the_plans_tests_are_translated_and_said_so() -> None:
    barkla = plan(test="ANOVA")
    barkla.statistical_tests += [
        barkla.statistical_tests[0].model_copy(update={"test": "Tukey’s HSD"}),
        barkla.statistical_tests[0].model_copy(update={"test": "Kruskal-Wallis"}),
    ]
    summaries, comparisons = aggregate(barkla, records())
    assert {c.test for c in comparisons} == {"paired t-test"}  # the first test decides: ANOVA -> t-tests
    assert all(c.p_adjusted is not None for c in comparisons)  # two comparisons: corrected although "none"
    assert correction_used(barkla, comparisons) == "holm"
    assert translated_tests(barkla) == ["ANOVA", "Tukey’s HSD", "Kruskal-Wallis"]
    notes = analysis_notes(barkla, comparisons)
    assert notes[0] == "each condition is compared with the baseline using the paired t-test"
    assert "the plan named ANOVA, Tukey’s HSD, Kruskal-Wallis; rtool runs pairwise t-tests" in notes[1]
    assert notes[2] == "p-values are corrected for the 2 comparisons with Holm's method"
    rank = aggregate(plan(test="Kruskal-Wallis H test"), records())[1]
    assert {c.test for c in rank} == {"Mann-Whitney U"}
    single = plan()
    single.metrics = single.metrics[:1]
    one = aggregate(single, records())[1]
    assert len(one) == 1 and one[0].p_adjusted is None and correction_used(single, one) == ""


def test_flat_results_are_not_called_significant() -> None:
    flat = [
        RunRecord(condition=c, seed=s, status="ok", metrics={"acc": v, "loss": 1 - v})
        for c, v in (("base", 1.0), ("new", 0.6))
        for s in (0, 1, 2)
    ]
    _, comparisons = aggregate(plan(), flat)
    assert all(c.p_value is None and not c.significant and c.better is None for c in comparisons)
    assert comparisons[0].verdict().startswith("no test possible: the results did not vary across seeds")
    assert analysis_notes(plan(), comparisons)[-1].startswith("no test was possible for acc, loss")


# --------------------------------------------------------------- playbook
@pytest.mark.parametrize(
    ("log", "signature"),
    [
        ("ModuleNotFoundError: No module named 'sklearn.linear_model'", "missing_module"),
        ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate", "cuda_oom"),
        ("slurmstepd: error: *** JOB 1 CANCELLED AT ... DUE TO TIME LIMIT ***", "time_limit"),
        ("rtool-contract: missing metric 'acc'", "contract"),
        ("  File 'experiment.py', line 3\nSyntaxError: invalid syntax", "syntax_error"),
        ('Traceback (most recent call last):\nKeyError: "label"', "traceback"),
        ("FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'", "file_not_found"),
    ],
)
def test_diagnose(log: str, signature: str) -> None:
    assert diagnose(log).signature == signature


def test_playbook_details() -> None:
    assert diagnose("No module named 'sklearn'").requirement == "scikit-learn"
    assert diagnose("", exit_code=137).signature == "cpu_oom"
    assert diagnose("Traceback (most recent call last):\nKeyError: 'label'").summary == "KeyError: 'label'"
    source = "# requirements: pandas, scikit-learn>=1.3, --index-url=http://evil, git+https://x/y\nBATCH_SIZE = 32\n"
    assert declared_requirements(source) == ["pandas", "scikit-learn>=1.3"]  # options and URLs never reach pip
    assert not valid_requirement("-r other.txt") and valid_requirement("torch-geometric")
    assert merge_requirements(["torch", "pandas"], ["pandas", "scipy"], provided=["torch"]) == ["pandas", "scipy"]
    assert "BATCH_SIZE = 16" in (halve_batch_size(source) or "")
    assert halve_batch_size("BATCH_SIZE = 1\n") is None and halve_batch_size("x = 1\n") is None


# ---------------------------------------------------------- code handling
GOOD = "def run(condition, seed, resources, smoke, workdir):\n    return {'acc': 1.0}\n"


def test_extract_code_and_static_checks(tmp_path: Path) -> None:
    reply = f"Here you go:\n```python\nprint('x')\n```\nand the file:\n```python\n{GOOD}```\n"
    assert extract_code(reply) == GOOD  # the longest block, i.e. the file
    assert extract_code("no code at all") is None
    path = tmp_path / "experiment.py"
    path.write_text(GOOD)
    assert static_problems(GOOD, path) == []
    assert static_problems("def run(:\n", path)[0].startswith("SyntaxError")
    assert "must accept" in static_problems("def run(condition, seed):\n    return {}\n", path)[0]
    if shutil.which("ruff") or Path(sys.executable).with_name("ruff").exists():
        undefined = "def run(condition, seed, resources, smoke, workdir):\n    return {'acc': missing_name}\n"
        path.write_text(undefined)
        assert any("missing_name" in p for p in static_problems(undefined, path))


def test_file_listing(tmp_path: Path) -> None:
    root = tmp_path / "resource"
    (root / "data").mkdir(parents=True)
    (root / "data" / "train.parquet").write_bytes(b"0" * 2_000_000)
    (root / ".cache").mkdir()
    (root / ".cache" / "x").write_text("hidden")
    assert file_listing(root, 10) == "data/train.parquet (2.0 MB)"
    assert file_listing(root, 0) == ""


def test_packs_load() -> None:
    assert {"generic", "ml_dl"} <= set(available_packs())
    pack = load_pack("ml_dl")
    assert "BATCH_SIZE" in pack.template and pack.hints and "torch" in pack.packages


# ---------------------------------------------------------------- results
def test_aggregate_and_render(tmp_path: Path) -> None:
    p = plan(correction="holm")
    summaries, comparisons = aggregate(p, records())
    acc = next(c for c in comparisons if c.metric == "acc")
    assert acc.test == "paired t-test" and acc.p_value == pytest.approx(0.00985, abs=1e-4)
    assert acc.significant and acc.better is True and acc.p_adjusted is not None
    loss = next(c for c in comparisons if c.metric == "loss")
    assert loss.difference < 0 and loss.better is True  # lower loss is better
    assert {(s.condition, s.metric) for s in summaries} == {(c, m) for c in ("base", "new") for m in ("acc", "loss")}
    payload = results_payload(records(), ["all good"])
    assert validate(payload, p.results_schema) == []
    assert validate({"runs": [{"condition": "x", "seed": 0, "metrics": {}}]}, p.results_schema)  # bad: caught
    figures = render_figures(p, summaries, tmp_path / "figures")
    assert set(figures) == {"acc", "loss"} and figures["acc"].read_text().startswith("<svg")
    table = render_tables(p, records(), summaries, comparisons, {"acc": "figures/acc.svg"})
    assert "| new vs base | 0.0667 |" in table and "![acc](figures/acc.svg)" in table
    wilcoxon = aggregate(plan(test="Wilcoxon signed-rank test"), records())[1][0]
    assert wilcoxon.test == "Wilcoxon signed-rank" and wilcoxon.p_value == pytest.approx(0.25)


def test_partial_results_still_aggregate() -> None:
    runs = records()
    runs[0] = RunRecord(condition="base", seed=0, status="failed", error="cuda_oom")
    summaries, comparisons = aggregate(plan(), runs)
    base = next(s for s in summaries if s.condition == "base" and s.metric == "acc")
    assert base.n == 2 and comparisons[0].test == "Welch t-test"  # seeds no longer pair up


def test_numbers_and_svg() -> None:
    assert [fmt(x) for x in (0.066666, 12345.6, 0.0, None, 1e-7)] == ["0.0667", "12346", "0", "n/a", "1.00e-07"]
    assert fmt_p(0.0004) == "< 0.001" and fmt_p(0.0312) == "0.031"
    assert nice_ticks(0, 0.74) == [0.0, 0.2, 0.4, 0.6, 0.8]
    svg = bar_chart("acc", ["a", "b"], [0.5, 0.7], [0.4, 0.6], [0.6, 0.8], highlight=0)
    assert svg.count("<rect") == 3 and "acc" in svg  # background + two bars


# ------------------------------------------------------------ environment
def test_container_commands_and_variables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIF", "/images/exp.sif")
    cfg = ExperimentsConfig(container="$SIF", home="/scratch/home", binds=["/mnt/scratch"], modules=["apptainer/1.3.6"])
    env = ExperimentEnv(cfg, tmp_path / "code")
    command = env.python(["run.py"], gpu=True)
    assert command[:3] == ["apptainer", "exec", "--nv"] and "/images/exp.sif" in command
    assert command[3:5] == ["--home", "/scratch/home"] and command[-2:] == ["python3", "run.py"]
    variables = env.variables(gpu=False)
    assert variables["CUDA_VISIBLE_DEVICES"] == "" and variables["APPTAINERENV_PYTHONNOUSERSITE"] == "1"
    compute = plan().compute
    spec = env.array_job(name="rtool-exp-code-1", workdir=tmp_path, batch_file="batches/full-1.json", tasks=6,
                         compute=compute)  # fmt: skip
    assert spec.array == "0-5%4" and spec.gpus is not None and spec.gpus.gres() == "gpu:l40s:1"
    assert '"$SIF"' in spec.command and '"$SLURM_ARRAY_TASK_ID"' in spec.command  # expanded by the job
    script = render_script(spec, sentinel_dir=tmp_path / "s", log_dir=tmp_path / "l")
    assert "module load apptainer/1.3.6" in script and "export PYTHONNOUSERSITE=1" in script


def test_empty_requirements_line_and_no_local_installs(tmp_path: Path) -> None:
    assert declared_requirements("# requirements:\nimport random\n") == []  # never reads the next line
    env = ExperimentEnv(ExperimentsConfig(), tmp_path)
    result = env.install(["surely-not-installed-pkg"], log_path=tmp_path / "install.log")
    assert result is not None and not result.ok and "not installing" in result.tail()
    assert not (tmp_path / ".venv").exists()  # no venv, no pip, no download on a machine without the container
    assert env.install(["json"], log_path=tmp_path / "ok.log").ok  # importable already: nothing to do


def test_local_runs_and_timeouts(tmp_path: Path) -> None:
    env = ExperimentEnv(ExperimentsConfig(), tmp_path)
    ok = env.run(["-c", "print('hello')"], log_path=tmp_path / "ok.log", timeout_s=30)
    assert ok.ok and "hello" in ok.tail()
    slow = env.run(["-c", "import time; time.sleep(30)"], log_path=tmp_path / "slow.log", timeout_s=0.5)
    assert slow.timed_out and not slow.ok and slow.seconds < 10


def test_harness_contract(tmp_path: Path) -> None:
    harness = resources.files("research_tool.agents.coder").joinpath("harness")
    for name in ("run.py", "launch.py"):
        (tmp_path / name).write_text(harness.joinpath(name).read_text())
    (tmp_path / "config.json").write_text(
        json.dumps({"conditions": {"a": {}}, "metrics": {"acc": {}}, "resources": {}, "seeds": [0]})
    )
    (tmp_path / "experiment.py").write_text(
        "import random\n"
        "def run(condition, seed, resources, smoke, workdir):\n"
        "    return {'acc': random.random(), 'extra': 'text'}\n"
    )
    out = tmp_path / "out" / "a.json"
    cmd = [sys.executable, "run.py", "--condition", "a", "--seed", "3", "--output", str(out)]
    assert subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True).returncode == 0
    first = json.loads(out.read_text())
    assert set(first["metrics"]) == {"acc"} and first["seed"] == 3  # seeded: deterministic
    subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True)
    assert math.isclose(json.loads(out.read_text())["metrics"]["acc"], first["metrics"]["acc"])
    (tmp_path / "experiment.py").write_text("def run(condition, seed, resources, smoke, workdir):\n    return {}\n")
    broken = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert broken.returncode == 3 and "rtool-contract: missing metric 'acc'" in broken.stdout
    batch = {
        "runs": [{"condition": "a", "seed": 1, "output": str(tmp_path / "b.json"), "workdir": str(tmp_path / "w")}]
    }
    (tmp_path / "batch.json").write_text(json.dumps(batch))
    launched = subprocess.run([sys.executable, "launch.py", "--batch", "batch.json", "--task", "0"], cwd=tmp_path)
    assert launched.returncode == 3  # the broken experiment again, now through the launcher
