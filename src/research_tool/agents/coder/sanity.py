"""Signs that experiment results may not come from a real experiment.

These are warnings for people: at the GPU-runs checkpoint, in the results, and as a failed Reviewer
check. Nothing here triggers an automatic repair, because a deterministic experiment can rightly give
the same number for every seed. Seen on Barkla (2026-10-12): an experiment.py returned
``1.0 - 0.1 * (n - 2)`` as its "success rate", under a comment calling it a mock. It ran 27 "GPU runs"
of 0 s each, and the result went into the report unchallenged.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections.abc import Sequence

from ...schemas.code import RunRecord
from ...schemas.plan import ExperimentPlan
from .stats import constant

_MOCK = re.compile(
    r"\b(mock(?:ed|ing|s)?|dumm(?:y|ies)|fake[ds]?|placeholders?|hypothetical|made[- ]up|for demonstration|"
    r"pretend(?:s|ed)?)\b",
    re.IGNORECASE,
)
FAST_S = 1.0  # every full run finished faster than this...
SLOW_PLAN_H = 10 / 60  # ...although the plan expects at least this long per run


def _quote(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def mock_admissions(source: str) -> list[tuple[int, str]]:
    """(line, text) of comments and in-function strings that call the code a mock, dummy, placeholder...

    The module docstring and module-level constants hold the plan's own words (the hypothesis, the
    condition descriptions), so only comments and strings inside functions count.
    """
    found: list[tuple[int, str]] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT and _MOCK.search(token.string):
                found.append((token.start[0], token.string.lstrip("# ")))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return found
    for function in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)):
        for node in ast.walk(function):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and _MOCK.search(node.value):
                found.append((node.lineno, node.value))
    return sorted(set(found))


def unused_names(source: str) -> list[str]:
    """Variables run() assigns but never reads (e.g. a data path it never opens, a loop count it never uses)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    run = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"), None)
    if run is None:
        return []
    assigned: dict[str, int] = {}
    read: set[str] = set()
    for node in ast.walk(run):
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name) and not name.id.startswith("_"):
                        assigned.setdefault(name.id, name.lineno)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            read.add(node.target.id)  # x += 1 reads x
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            read.add(node.id)
    return [name for name, _line in sorted(assigned.items(), key=lambda item: item[1]) if name not in read]


def reads_resources(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return True
    run = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"), None)
    return run is None or any(
        isinstance(n, ast.Name) and n.id == "resources" and isinstance(n.ctx, ast.Load) for n in ast.walk(run)
    )


def code_warnings(source: str, *, has_resources: bool) -> list[str]:
    """What a person should look at in experiment.py before approving GPU time for it."""
    warnings = []
    admissions = mock_admissions(source)
    if admissions:
        where = "; ".join(f"line {line}: “{_quote(text)}”" for line, text in admissions[:3])
        warnings.append(f"experiment.py calls part of itself a mock, dummy or placeholder ({where})")
    unused = unused_names(source)
    if unused:
        shown = ", ".join(f"`{n}`" for n in unused[:6]) + (f" and {len(unused) - 6} more" if len(unused) > 6 else "")
        them = "it" if len(unused) == 1 else "them"
        warnings.append(f"run() sets {shown} but never uses {them}: is part of the procedure missing?")
    if has_resources and not reads_resources(source):
        warnings.append("run() never reads `resources`, although data or models were downloaded for it")
    return warnings


def seed_warning(condition: str, seeds: tuple[int, int], first: dict[str, float], second: dict[str, float]) -> str:
    """'' unless two smoke runs of one condition with different seeds gave exactly the same metrics."""
    shared = [m for m in first if m in second]
    if shared and all(constant([first[m], second[m]]) for m in shared):
        return (
            f"the smoke test of `{condition}` gave exactly the same metrics with seeds {seeds[0]} and {seeds[1]}: "
            "the results may not depend on the seed (is anything random actually used?)"
        )
    return ""


def missing_resources_warning(names: Sequence[str]) -> str:
    if not names:
        return ""
    listed = ", ".join(names)
    return (
        f"{listed} {'was' if len(names) == 1 else 'were'} never provided (the folder is empty), so experiment.py "
        f"had to build {'it' if len(names) == 1 else 'them'} itself: check that this is real"
    )


def results_warnings(plan: ExperimentPlan, records: Sequence[RunRecord]) -> list[str]:
    """What the finished runs say about whether anything was really measured."""
    ok = [r for r in records if r.status == "ok"]
    warnings = []
    flat = []
    for metric in plan.metrics:
        groups = [
            [r.metrics[metric.name] for r in ok if r.condition == c.name and metric.name in r.metrics]
            for c in plan.run_conditions()
        ]
        groups = [g for g in groups if len(g) >= 2]
        if groups and all(constant(g) for g in groups):
            flat.append(f"`{metric.name}`")
    if flat:
        warnings.append(
            f"every seed gave exactly the same {', '.join(flat)} in every condition: the seeds had no effect, so "
            "confidence intervals and tests are not meaningful (is the value computed rather than measured?)"
        )
    seconds = [r.seconds for r in ok if r.seconds is not None]
    if seconds and max(seconds) < FAST_S and plan.compute.hours_per_run >= SLOW_PLAN_H:
        warnings.append(
            f"every run finished in under {FAST_S:g} s, although the plan expects about "
            f"{plan.compute.hours_per_run:g} h per run: did the training or simulation really run?"
        )
    return warnings
