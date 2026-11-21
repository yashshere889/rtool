"""The study's data: waiting for the upload, checking it against the protocol, and its analysis.

``collect_data`` waits at the ``data_upload`` checkpoint until the researcher has put data.csv in the
stage's inbox (``rtool add-data``), or finds it among the run's local data (``study_data``, as validity
cases provide it). A file that does not match the protocol stops the stage with every problem listed.

``analyse_table`` analyses the table with rtool's own script (written into the run, so it can be read
and rerun, and audited like any experiment): one entry per unit and condition, compared across units
as every data analysis is (``results.aggregate``, units mode).
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

from ...core.agent import Task
from ...core.context import AgentContext
from ...core.errors import StageFailed
from ...core.fs import atomic_write_json, atomic_write_text, sha256_file, sha256_text
from ...core.registry import register_agent
from ...packs import load_pack, pack_result_issues
from ...schemas.code import CodeBundle, CodeFile, ExperimentResults, RunRecord
from ...schemas.common import ResearchBrief
from ...schemas.plan import ExperimentPlan
from ..coder import sanity
from ..coder.harness.run import check_units, unit_record
from ..coder.results import (
    aggregate,
    analysis_notes,
    confounding_issues,
    correction_used,
    render_figures,
    render_tables,
    results_payload,
    variation_notes,
)
from .schemas import DATA_FILE, Protocol, StudyData

GATE = "data_upload"


def read_table(path: Path, plan: ExperimentPlan) -> tuple[int, dict[str, int], list[str]]:
    """Rows, units per condition, and every way the table breaks the protocol's format."""
    metrics = [m.name for m in plan.metrics]
    conditions = [c.name for c in plan.run_conditions()]
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    units = dict.fromkeys(conditions, 0)
    rows = 0
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = [c for c in ("unit", "condition", *metrics) if c not in (reader.fieldnames or [])]
            if missing:
                return 0, units, [f"the header lacks the columns {', '.join(missing)} (it has {reader.fieldnames})"]
            for line, row in enumerate(reader, start=2):
                rows += 1
                unit, condition = (row.get("unit") or "").strip(), (row.get("condition") or "").strip()
                if not unit:
                    problems.append(f"line {line}: no unit id")
                if condition not in units:
                    problems.append(f"line {line}: condition {condition!r} is not one of {', '.join(conditions)}")
                    continue
                if (unit, condition) in seen:
                    problems.append(f"line {line}: unit {unit} appears twice under {condition}")
                seen.add((unit, condition))
                for name in metrics:
                    try:
                        value = float(row.get(name) or "nan")
                    except ValueError:
                        value = math.nan
                    if not math.isfinite(value):
                        problems.append(f"line {line}: {name} is {row.get(name)!r}, not a number")
                units[condition] += 1
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return rows, units, [f"{path.name} cannot be read as CSV: {exc}"]
    problems += [f"condition {c} has {n} unit(s); at least 2 are needed" for c, n in units.items() if n < 2]
    return rows, units, problems


@register_agent("collect_data")
class CollectData(Task):
    description = "Waits for the study's data.csv, and checks it against the protocol."
    consumes = ("experiment_plan", "protocol")
    produces = ("study_data",)

    def find(self, ctx: AgentContext) -> Path | None:
        provided = ctx.app_config.experiments.local_data.get("study_data")
        if provided is not None:
            path = Path(provided.path).expanduser()
            path = path / DATA_FILE if path.is_dir() else path
            if path.is_file():
                return path
        uploaded = ctx.workspace.path("inbox") / DATA_FILE
        return uploaded if uploaded.is_file() else None

    def run(self, ctx: AgentContext) -> None:
        plan: ExperimentPlan = ctx.inputs.get("experiment_plan")
        protocol: Protocol = ctx.inputs.get("protocol")
        inbox = ctx.workspace.path("inbox")
        source = self.find(ctx)
        if source is None:
            inbox.mkdir(parents=True, exist_ok=True)
            decision = ctx.gate(
                GATE,
                {
                    "message": f"Carry out the study as the protocol says ({protocol.markdown_path}), then upload "
                    f"its data with `rtool add-data <run> data.csv` (it goes to {inbox / DATA_FILE}). "
                    f"{protocol.data_format}.",
                    "inbox": str(inbox),
                },
            )
            if not decision.approved:
                raise StageFailed(
                    f"the data checkpoint was rejected: {decision.feedback or 'no reason given'}", needs_human=True
                )
            source = self.find(ctx)
            if source is None:
                raise StageFailed(
                    f"no {DATA_FILE} was uploaded: put it in {inbox} (rtool add-data), then rtool retry",
                    needs_human=True,
                )
        rows, units, problems = read_table(source, plan)
        if problems:
            shown = "; ".join(problems[:8]) + (f"; and {len(problems) - 8} more" if len(problems) > 8 else "")
            raise StageFailed(
                f"{DATA_FILE} does not match the protocol: {shown}. Fix the file, upload it again, then rtool retry",
                needs_human=True,
            )
        copy = ctx.workspace.path(DATA_FILE)
        if source.resolve() != copy.resolve():
            shutil.copy2(source, copy)
        ctx.emit("study_data_received", rows=rows, units=units)
        ctx.outputs.put(
            "study_data", StudyData(path=ctx.relative(copy), sha256=sha256_file(copy), rows=rows, units=units)
        )


SCRIPT = '''"""rtool's analysis of the study's data.csv, written by rtool from the protocol (do not edit).

    run(condition, ...) -> {{"units": [...]}}: one entry per unit measured under that condition.
"""

import csv
from pathlib import Path

METRICS = {metrics!r}


def run(condition, seed, resources, smoke, workdir):
    units = []
    with open(Path(resources["study_data"]) / "data.csv", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["condition"].strip() == condition:
                entry = {{"unit": row["unit"].strip(), "batch": (row.get("batch") or "").strip()}}
                entry.update({{name: float(row[name]) for name in METRICS}})
                units.append(entry)
    return {{"units": units}}
'''


@register_agent("analyse_table")
class AnalyseTable(Task):
    description = "Analyses the uploaded study data across units, with rtool's own script."
    consumes = ("research_brief", "experiment_plan", "study_data")
    produces = ("code_bundle", "experiment_results")

    def run(self, ctx: AgentContext) -> None:
        brief: ResearchBrief = ctx.inputs.get("research_brief")
        plan: ExperimentPlan = ctx.inputs.get("experiment_plan")
        data: StudyData = ctx.inputs.get("study_data")
        metrics = [m.name for m in plan.metrics]
        code = ctx.workspace.mkdir("code")
        script = SCRIPT.format(metrics=metrics)
        atomic_write_text(code / "experiment.py", script)
        resources = {"study_data": str(ctx.absolute(data.path).parent)}
        config = {"conditions": [c.name for c in plan.run_conditions()], "metrics": metrics, "resources": resources}
        atomic_write_json(code / "config.json", {**config, "replicates": "units"})
        namespace: dict[str, Any] = {}
        exec(compile(script, str(code / "experiment.py"), "exec"), namespace)  # noqa: S102 - rtool's own script, above
        records: list[RunRecord] = []
        for condition in config["conditions"]:
            result = namespace["run"](condition, 0, resources, False, str(code))
            problems = check_units(result, metrics)
            if problems:
                raise StageFailed(
                    f"the data of {condition} cannot be analysed: {'; '.join(problems)}", needs_human=True
                )
            _, units = unit_record(result["units"], metrics)
            records += [
                RunRecord(
                    condition=condition, seed=0, status="ok", unit=u["unit"], batch=u["batch"], metrics=u["metrics"]
                )
                for u in units
            ]
        summaries, comparisons = aggregate(plan, records)
        warnings = sanity.results_warnings(plan, records)
        issues = confounding_issues(plan, records) + pack_result_issues(load_pack(brief.domain_pack), plan, records)
        notes = analysis_notes(plan, comparisons) + variation_notes(plan, records, comparisons)
        notes += [f"Warning: {w}" for w in warnings] + [f"Design problem: {i}" for i in issues]
        out = ctx.workspace.mkdir("results")
        results_json = out / "results.json"
        atomic_write_text(results_json, json.dumps(results_payload(records, notes), indent=2) + "\n")
        figures = render_figures(plan, summaries, out / "figures")
        tables = out / "results.md"
        figure_links = {m: f"figures/{p.name}" for m, p in figures.items()}
        atomic_write_text(tables, render_tables(plan, records, summaries, comparisons, figure_links, warnings=warnings))
        files = [
            CodeFile(path=name, sha256=sha256_text(text), lines=text.count("\n"), origin="harness")
            for name, text in (("experiment.py", script), ("config.json", (code / "config.json").read_text()))
        ]
        fingerprint = sha256_text("\n".join(f"{f.path}:{f.sha256}" for f in files))
        ctx.outputs.put(
            "code_bundle",
            CodeBundle(root=ctx.relative(code), files=files, pack=brief.domain_pack, fingerprint=fingerprint),
        )
        ctx.outputs.put(
            "experiment_results",
            ExperimentResults(
                runs=records,
                summaries=summaries,
                comparisons=comparisons,
                complete=True,
                results_path=ctx.relative(results_json),
                tables_path=ctx.relative(tables),
                figures=[ctx.relative(p) for p in figures.values()],
                code_fingerprint=fingerprint,
                notes=notes,
                warnings=warnings,
                issues=issues,
                correction=correction_used(plan, comparisons),
            ),
            meta={"review": ctx.relative(tables)},
        )
