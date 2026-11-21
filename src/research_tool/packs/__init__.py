"""Domain packs: field-specific prompts, code skeletons and hints, without touching agent code.

A pack is a folder ``packs/<name>/`` with:

* ``pack.yaml`` — description, hints for the Coder, what "smoke mode" means, usual packages;
* ``templates/experiment.py.j2`` — the skeleton the Coder starts from;
* ``prompts/<agent>/*.j2`` — optional prompt overrides (found by the prompt library);
* in ``pack.yaml``, optionally: ``plan_checks`` (``"module:function"`` names of checks run on every
  plan, ``(plan, brief) -> list[str]``), ``result_checks`` (the same for finished runs,
  ``(plan, records) -> list[str]``, reported as ``results.issues``) and ``resource_sources`` (the only
  sources its plans may use).

The research brief names the pack (``rtool new --pack ml_dl``).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING

import yaml

from ..core.errors import ConfigError

if TYPE_CHECKING:
    from ..schemas.code import RunRecord
    from ..schemas.common import ResearchBrief
    from ..schemas.plan import ExperimentPlan

PlanCheck = Callable[["ExperimentPlan", "ResearchBrief | None"], list[str]]

DEFAULT_PACK = "generic"


@dataclass
class Pack:
    name: str
    description: str
    smoke: str  # what a smoke run must do, in one sentence
    hints: list[str] = field(default_factory=list)
    packages: list[str] = field(default_factory=list)  # what this kind of experiment usually imports
    template: str = ""  # Jinja source of experiment.py
    plan_checks: list[str] = field(default_factory=list)  # "module:function", each (plan, brief) -> issues
    result_checks: list[str] = field(default_factory=list)  # "module:function", each (plan, records) -> issues
    resource_sources: list[str] = field(default_factory=list)  # the sources its plans may use; empty: any
    replicates: str = "seeds"  # "units": its plans analyse data, compared across units (donors, samples)
    review_checks: list[str] = field(default_factory=list)  # questions the Reviewer asks of every study
    literature_sources: list[str] = field(default_factory=list)  # search sources for its topics; empty: config's


def available_packs() -> list[str]:
    root = resources.files(__name__)
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.joinpath("pack.yaml").is_file())


def load_pack(name: str) -> Pack:
    folder = resources.files(__name__).joinpath(name)
    if not folder.joinpath("pack.yaml").is_file():
        raise ConfigError(f"unknown domain pack {name!r}; available: {available_packs()}")
    data = yaml.safe_load(folder.joinpath("pack.yaml").read_text(encoding="utf-8")) or {}
    template = folder.joinpath("templates", "experiment.py.j2")
    pack = Pack(
        name=name,
        description=str(data.get("description", "")),
        smoke=str(data.get("smoke", "In smoke mode use a tiny problem so the run takes seconds on a CPU.")),
        hints=[str(h) for h in data.get("hints", [])],
        packages=[str(p) for p in data.get("packages", [])],
        template=template.read_text(encoding="utf-8") if template.is_file() else "",
        plan_checks=[str(c) for c in data.get("plan_checks", [])],
        result_checks=[str(c) for c in data.get("result_checks", [])],
        resource_sources=[str(s) for s in data.get("resource_sources", [])],
        replicates=str(data.get("replicates", "seeds")),
        review_checks=[str(q) for q in data.get("review_checks", [])],
        literature_sources=[str(s) for s in data.get("literature_sources", [])],
    )
    if pack.replicates not in ("seeds", "units"):
        raise ConfigError(f"pack {name!r}: replicates must be 'seeds' or 'units', not {pack.replicates!r}")
    for ref in [*pack.plan_checks, *pack.result_checks]:
        _plan_check(name, ref)  # a misspelt check fails when the pack is chosen, not after the plan is written
    return pack


def _plan_check(pack: str, ref: str) -> PlanCheck:
    module, _, function = ref.partition(":")
    try:
        check = getattr(importlib.import_module(module), function)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ConfigError(f"pack {pack!r}: plan check {ref!r} cannot be loaded: {exc}") from None
    if not callable(check):
        raise ConfigError(f"pack {pack!r}: plan check {ref!r} is not a function")
    return check


def pack_plan_issues(pack: Pack, plan: ExperimentPlan, brief: ResearchBrief | None) -> list[str]:
    """What the pack's own checks find in a plan."""
    return [issue for ref in pack.plan_checks for issue in _plan_check(pack.name, ref)(plan, brief)]


def pack_result_issues(pack: Pack, plan: ExperimentPlan, records: Sequence[RunRecord]) -> list[str]:
    """What the pack's own checks find in finished runs."""
    return [issue for ref in pack.result_checks for issue in _plan_check(pack.name, ref)(plan, records)]
