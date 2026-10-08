"""Versioned Jinja2 prompt templates.

Templates live next to the agent that uses them (``agents/<agent>/prompts/*.j2``).
A domain pack (``packs/<pack>/prompts/<agent>/``) or any extra directory can
override a template by providing a file with the same name; the first match in
the search path wins. Rendering is strict (an undefined variable is an error)
and returns the template hash, which agents store in artifact metadata so every
output records exactly which prompt produced it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from jinja2 import ChoiceLoader, Environment, FileSystemLoader, StrictUndefined, TemplateNotFound, UndefinedError

from ..core.errors import RToolError
from ..core.fs import sha256_text


class PromptError(RToolError):
    """A prompt template is missing or failed to render."""


@dataclass(frozen=True)
class RenderedPrompt:
    name: str
    text: str
    source: str  # file the template came from
    template_hash: str

    def provenance(self) -> dict[str, str]:
        return {"prompt": self.name, "prompt_hash": self.template_hash, "prompt_source": self.source}


def _package_dir(package: str, *parts: str) -> Path | None:
    try:
        candidate = resources.files(package).joinpath(*parts)
    except ModuleNotFoundError:
        return None
    path = Path(str(candidate))
    return path if path.is_dir() else None


class PromptLibrary:
    def __init__(self, search_paths: Iterable[Path]) -> None:
        self.search_paths = [Path(p) for p in search_paths if Path(p).is_dir()]
        self.env = Environment(
            loader=ChoiceLoader([FileSystemLoader(str(p)) for p in self.search_paths]),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        )

    @classmethod
    def for_agent(
        cls,
        agent_package: str,
        agent_name: str,
        *,
        pack: str | None = None,
        extra_dirs: Iterable[Path] = (),
    ) -> PromptLibrary:
        """Search order: extra dirs, then the pack's overrides, then the agent's own prompts."""
        paths: list[Path] = [Path(d) / agent_name for d in extra_dirs]
        if pack:
            pack_dir = _package_dir("research_tool.packs", pack, "prompts", agent_name)
            if pack_dir:
                paths.append(pack_dir)
        own = _package_dir(agent_package, "prompts")
        if own:
            paths.append(own)
        return cls(paths)

    def names(self) -> list[str]:
        return sorted(self.env.list_templates())

    def render(self, name: str, **variables: Any) -> RenderedPrompt:
        try:
            template = self.env.get_template(name)
        except TemplateNotFound:
            raise PromptError(f"prompt {name!r} not found in {[str(p) for p in self.search_paths]}") from None
        source_path = template.filename or name
        source_text = Path(source_path).read_text(encoding="utf-8") if template.filename else ""
        try:
            text = template.render(**variables)
        except UndefinedError as exc:
            raise PromptError(f"prompt {name!r}: {exc}") from None
        return RenderedPrompt(
            name=name, text=text.strip() + "\n", source=str(source_path), template_hash=sha256_text(source_text)
        )
