"""Domain packs: field-specific prompts, code skeletons and hints, without touching agent code.

A pack is a folder ``packs/<name>/`` with:

* ``pack.yaml`` — description, hints for the Coder, what "smoke mode" means, usual packages;
* ``templates/experiment.py.j2`` — the skeleton the Coder starts from;
* ``prompts/<agent>/*.j2`` — optional prompt overrides (found by the prompt library).

The research brief names the pack (``rtool new --pack ml_dl``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources

import yaml

from ..core.errors import ConfigError

DEFAULT_PACK = "generic"


@dataclass
class Pack:
    name: str
    description: str
    smoke: str  # what a smoke run must do, in one sentence
    hints: list[str] = field(default_factory=list)
    packages: list[str] = field(default_factory=list)  # what this kind of experiment usually imports
    template: str = ""  # Jinja source of experiment.py


def available_packs() -> list[str]:
    root = resources.files(__name__)
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.joinpath("pack.yaml").is_file())


def load_pack(name: str) -> Pack:
    folder = resources.files(__name__).joinpath(name)
    if not folder.joinpath("pack.yaml").is_file():
        raise ConfigError(f"unknown domain pack {name!r}; available: {available_packs()}")
    data = yaml.safe_load(folder.joinpath("pack.yaml").read_text(encoding="utf-8")) or {}
    template = folder.joinpath("templates", "experiment.py.j2")
    return Pack(
        name=name,
        description=str(data.get("description", "")),
        smoke=str(data.get("smoke", "In smoke mode use a tiny problem so the run takes seconds on a CPU.")),
        hints=[str(h) for h in data.get("hints", [])],
        packages=[str(p) for p in data.get("packages", [])],
        template=template.read_text(encoding="utf-8") if template.is_file() else "",
    )
