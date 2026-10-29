from __future__ import annotations

import contextlib
import json
from pathlib import Path

from typer.testing import CliRunner

from research_tool.cli.main import app

runner = CliRunner()


def invoke(*args: object) -> tuple[int, str]:
    result = runner.invoke(app, [str(a) for a in args])
    text = result.output
    with contextlib.suppress(ValueError):  # stderr is mixed into output on some Click versions
        text += result.stderr
    return result.exit_code, text


def test_toy_pipeline_through_the_cli(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    code, out = invoke("new", "demo", "--topic", "sleep", "--pipeline", "toy", "--runs-root", root, "--run-id", "r1")
    assert code == 0, out
    run_dir = root / "demo" / "r1"
    code, out = invoke("run", run_dir, "--simulate-jobs")
    assert code == 0 and "waiting_gate" in out and "numbers_review" in out
    for _ in range(8):  # numbers_review, gpu_runs, gpu_runs (after the revision), final
        code, out = invoke("approve", run_dir, "--now", "--simulate-jobs")
        assert code == 0, out
        if "done: all stages are done" in out:
            break
    assert "done: all stages are done" in out
    code, out = invoke("status", run_dir)
    assert code == 0 and "toy_report v" in out
    assert '"mentions_variance": true' in invoke("show", run_dir, "toy_report")[1]
    assert "toy_numbers" in invoke("show", run_dir)[1]
    assert '"stage_finished"' in invoke("events", run_dir, "--kind", "stage_finished")[1]
    code, out = invoke("reopen", run_dir, "--stage", "source", "--now", "--simulate-jobs")
    assert code == 0 and "numbers_review" in out, out
    code, out = invoke("approve", run_dir, "--stage", "source", "--now", "--simulate-jobs")
    assert code == 0 and "done: all stages are done" in out, out  # approved as it was: nothing runs again


def test_select_and_errors_through_the_cli(tmp_path: Path) -> None:
    code, out = invoke("status", "missing-run", "--runs-root", tmp_path)
    assert code == 2 and "no run found" in out
    code, out = invoke("new", "demo", "--topic", "t", "--pipeline", "nope", "--runs-root", tmp_path)
    assert code == 2 and "no pipeline 'nope'" in out
    code, out = invoke("new", "demo", "--topic", "t", "--run-id", "../escape", "--runs-root", tmp_path)
    assert code == 2 and "invalid run id" in out


def test_agents_and_pipelines_commands() -> None:
    assert "toy_source" in invoke("agents")[1]
    code, out = invoke("pipelines", "--check", "toy")
    assert code == 0 and "numbers_review" in out and "analyze" in out
    assert "toy" in invoke("pipelines")[1]


def test_new_checks_the_domain_pack(tmp_path: Path) -> None:
    code, out = invoke("new", "demo", "--topic", "t", "--pack", "ml-dl", "--runs-root", tmp_path)
    assert code == 2 and "unknown domain pack" in out and "ml_dl" in out


def test_archive_keeps_results_and_leaves_out_rebuildable_files(tmp_path: Path) -> None:
    import tarfile

    root = tmp_path / "runs"
    code, out = invoke("new", "demo", "--topic", "t", "--pipeline", "toy", "--runs-root", root, "--run-id", "r1")
    assert code == 0, out
    run_dir = root / "demo" / "r1"
    (run_dir / "work" / "code" / ".venv" / "lib").mkdir(parents=True)
    (run_dir / "work" / "code" / ".venv" / "lib" / "big.so").write_bytes(b"0" * 10)
    (run_dir / "work" / "code" / "huge.bin").write_bytes(b"0" * 2_000_000)
    (run_dir / "work" / "code" / "report.md").write_text("# result")
    code, out = invoke("archive", run_dir, "--to", tmp_path / "data", "--max-file-mb", "1")
    assert code == 0, out
    assert "left out (too large): work/code/huge.bin" in out
    with tarfile.open(tmp_path / "data" / "demo-r1.tar.gz") as tar:
        names = tar.getnames()
        manifest = json.loads(tar.extractfile("demo-r1/MANIFEST.json").read())  # type: ignore[union-attr]
    assert "demo-r1/work/code/report.md" in names and "demo-r1/run.json" in names
    assert not any(".venv" in n or "huge.bin" in n for n in names)
    assert manifest["skipped_large_files"][0]["path"] == "work/code/huge.bin"


def test_a_project_name_means_its_latest_run(tmp_path: Path) -> None:
    import os
    import time

    root = tmp_path / "runs"
    for run_id in ("first", "second"):
        code, out = invoke("new", "demo", "--topic", "t", "--pipeline", "toy", "--runs-root", root, "--run-id", run_id)
        assert code == 0, out
    old = time.time() - 60
    os.utime(root / "demo" / "first" / "run.json", (old, old))
    code, out = invoke("status", "demo", "--runs-root", root)
    assert code == 0 and "demo/second" in out


def test_editing_the_plan_through_the_cli(tmp_path: Path) -> None:
    from research_tool.core.actions import ActionInbox
    from research_tool.core.artifacts import ArtifactStore
    from research_tool.testing.demo_run import build_demo_run

    run = build_demo_run(tmp_path / "runs", "experiment_plan")
    plan = ArtifactStore(run.artifacts_dir).get("experiment_plan").model_dump(mode="json")
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    code, out = invoke("edit", run.root, "experiment_plan", "--from", source)
    assert code != 0 and "nothing to change" in out
    source.write_text(json.dumps(plan | {"metrics": []}))
    code, out = invoke("edit", run.root, "experiment_plan", "--from", source)
    assert code != 0 and "at least one metric" in out
    source.write_text(json.dumps(plan | {"expected_results": "a bigger gain"}))
    code, out = invoke("edit", run.root, "experiment_plan", "--from", source, "--note", "sharper")
    assert code == 0 and "changes: expected_results" in out, out
    [(_, action)] = ActionInbox(run.actions_dir).pending()
    assert action.kind.value == "edit" and action.stage == "plan" and action.payload["base_version"] == 1
