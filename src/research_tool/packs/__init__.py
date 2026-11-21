"""Domain packs: field-specific prompts, code skeletons and hints, without touching agent code.

A pack is a folder with:

* ``pack.yaml`` — a title, a description, hints for the Coder, what "smoke mode" means, usual packages;
* ``templates/experiment.py.j2`` — the skeleton the Coder starts from;
* ``prompts/<agent>/*.j2`` — optional prompt overrides (found by the prompt library);
* in ``pack.yaml``, optionally: ``plan_checks`` (``"module:function"`` names of checks run on every
  plan, ``(plan, brief) -> list[str]``), ``result_checks`` (the same for finished runs,
  ``(plan, records) -> list[str]``, reported as ``results.issues``), ``resource_sources`` (the only
  sources its plans may use), ``pipeline`` (the pipeline its runs use unless the researcher picks
  one) and ``refuse`` (rules added to what rtool refuses in every pack, see ``refusals.py``).

The built-in packs are the folders here. Others come from installed Python packages: an entry point
in the group ``research_tool.packs`` names the package whose folder holds the pack. An installed pack
never replaces a built-in one, so a validity card always means the pack it names.

The research brief names the pack (``rtool new --pack ml_dl``); ``rtool packs`` lists them.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib import metadata, resources
from importlib.resources.abc import Traversable
from typing import TYPE_CHECKING

import yaml

from ..core.errors import ConfigError
from .refusals import Refusal, parse_refusals, refusal_reason

if TYPE_CHECKING:
    from ..schemas.code import RunRecord
    from ..schemas.common import ResearchBrief
    from ..schemas.plan import ExperimentPlan

PlanCheck = Callable[["ExperimentPlan", "ResearchBrief | None"], list[str]]

DEFAULT_PACK = "generic"
ENTRY_POINTS = "research_tool.packs"  # installed packages name their pack folders in this group
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")  # one path segment: names come from briefs and the web


@dataclass
class Pack:
    name: str
    description: str
    smoke: str  # what a smoke run must do, in one sentence
    title: str = ""  # a short name for menus (the web UI's Field menu)
    origin: str = "built-in"  # or the installed package it comes from, with its version
    hints: list[str] = field(default_factory=list)
    packages: list[str] = field(default_factory=list)  # what this kind of experiment usually imports
    template: str = ""  # Jinja source of experiment.py
    plan_checks: list[str] = field(default_factory=list)  # "module:function", each (plan, brief) -> issues
    result_checks: list[str] = field(default_factory=list)  # "module:function", each (plan, records) -> issues
    resource_sources: list[str] = field(default_factory=list)  # the sources its plans may use; empty: any
    replicates: str = "seeds"  # "units": its plans analyse data, compared across units (donors, samples)
    review_checks: list[str] = field(default_factory=list)  # questions the Reviewer asks of every study
    literature_sources: list[str] = field(default_factory=list)  # search sources for its topics; empty: config's
    refusals: tuple[Refusal, ...] = ()  # its own rules; the defaults in refusals.py apply to every pack
    pipeline: str = ""  # the pipeline its runs use unless the researcher picks one; "": research


@dataclass(frozen=True)
class PackSource:
    """Where a pack comes from: built in, or an installed package (an entry point names its module)."""

    name: str
    origin: str  # "built-in", or the installed package's name and version
    module: str = ""  # an installed pack's module: its folder holds pack.yaml
    shadowed: bool = False  # an installed pack whose name another pack has: listed, never used


def pack_sources() -> list[PackSource]:
    """Every pack, the built-in ones first. A name stays with the first pack that has it."""
    root = resources.files(__name__)
    found = sorted(p.name for p in root.iterdir() if p.is_dir() and p.joinpath("pack.yaml").is_file())
    sources = [PackSource(name, "built-in") for name in found]
    taken = set(found)
    for entry in sorted(metadata.entry_points(group=ENTRY_POINTS), key=lambda e: (e.name, e.value)):
        origin = f"{entry.dist.name} {entry.dist.version}" if entry.dist is not None else entry.value
        sources.append(PackSource(entry.name, origin, entry.module, shadowed=entry.name in taken))
        taken.add(entry.name)
    return sources


def available_packs() -> list[str]:
    return sorted(s.name for s in pack_sources() if not s.shadowed)


def _find(name: str) -> tuple[Traversable, str] | None:
    """The pack's folder and where it comes from, or None for an unknown pack."""
    if not _NAME.fullmatch(name):
        return None
    built_in = resources.files(__name__).joinpath(name)
    if built_in.joinpath("pack.yaml").is_file():
        return built_in, "built-in"
    source = next((s for s in pack_sources() if s.name == name and not s.shadowed), None)
    if source is None:
        return None
    try:
        return resources.files(source.module), source.origin
    except (ModuleNotFoundError, TypeError) as exc:
        raise ConfigError(f"pack {name!r} ({source.origin}) cannot be imported: {exc}") from None


def pack_folder(name: str) -> Traversable | None:
    """The folder of the pack's files (prompts, templates), or None for an unknown pack."""
    found = _find(name)
    return found[0] if found else None


def require_offered(name: str, allowed: Sequence[str]) -> None:
    """A deployment's ``packs.allowed``: new runs may use only those packs (empty: every pack)."""
    if allowed and name not in allowed:
        raise ConfigError(f"the {name!r} pack is not offered here; this deployment offers: {', '.join(allowed)}")


def load_pack(name: str) -> Pack:
    found = _find(name)
    if found is None:
        raise ConfigError(f"unknown domain pack {name!r}; available: {available_packs()}")
    folder, origin = found
    if not folder.joinpath("pack.yaml").is_file():
        raise ConfigError(f"pack {name!r} ({origin}) has no pack.yaml in its folder")
    try:
        data = yaml.safe_load(folder.joinpath("pack.yaml").read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"pack {name!r} ({origin}): pack.yaml is not valid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"pack {name!r} ({origin}): pack.yaml must be a mapping")
    template = folder.joinpath("templates", "experiment.py.j2")
    pack = Pack(
        name=name,
        title=str(data.get("title") or name),
        origin=origin,
        pipeline=str(data.get("pipeline") or ""),
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
        refusals=parse_refusals(name, data.get("refuse")),
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


def pack_refusal(pack: Pack, *texts: str) -> str | None:
    """Why rtool refuses these texts in this pack (the default rules and the pack's own), or None."""
    return refusal_reason(texts, pack.refusals)
