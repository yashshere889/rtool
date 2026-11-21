"""What rtool refuses in every pack (docs/roadmap.md, "Rules that apply to every stage").

* Work that makes a pathogen more dangerous (gain of function: more transmissible, virulent or
  lethal), and the design of toxins, toxic proteins or chemical and biological weapons. Refused when
  the run is created (the brief) and at the plan (the selected hypothesis and the plan's objective).
* Controlled-access human data (dbGaP, EGA). The resource check reports it, so the planner chooses
  again; the fetch refuses it; a brief that names it is refused.

A pack may add rules of its own (``refuse:`` in pack.yaml); the defaults always apply.

shortcut: regular expressions over the text of the brief and the plan. They stop requests made in
plain words, not someone who rephrases to get past them; a classifier would do better, if a
deployment needs more than this.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..core.errors import ConfigError


@dataclass(frozen=True)
class Refusal:
    """Refuse a text in which every pattern of one of ``rules`` matches, unless ``unless`` matches."""

    reason: str
    rules: tuple[tuple[re.Pattern[str], ...], ...]
    unless: re.Pattern[str] | None = None  # a purpose that makes the same words harmless (an antitoxin)

    def matches(self, text: str) -> bool:
        if self.unless is not None and self.unless.search(text):
            return False
        return any(all(p.search(text) for p in rule) for rule in self.rules)


def _re(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_NEAR = r"(?:\W+\w+){0,6}?\W+"  # up to six words in between
_PATHOGEN = _re(
    r"\b(virus(es)?|viral|virions?|pathogens?|pathogenic|influenza|coronavirus(es)?|sars|mers|h5n1|h7n9|ebola|"
    r"marburg|smallpox|variola|anthrax|bacillus anthracis|plague|yersinia|bacteri(a|um|al))\b"
)
_MAKE = (
    r"(design|designing|engineer|engineering|create|creating|generate|generating|make|making|produce|producing|"
    r"synthesi[sz]e|synthesi[sz]ing|optimi[sz]e|optimi[sz]ing|improve|improving|enhance|enhancing)"
)
_RAISE = (
    r"(enhance|enhancing|increase|increasing|boost|boosting|maximi[sz]e|maximi[sz]ing|raise|raising|amplify|"
    r"amplifying|improve|improving)"
)
_DANGER = (
    r"(transmissibility|transmission|virulence|pathogenicity|lethality|infectivity|host range|immune (escape|evasion))"
)
_WEAPON = (
    r"(toxins?|toxic proteins?|venom (proteins?|peptides?)|nerve agents?|chemical weapons?|biological weapons?|"
    r"bioweapons?)"
)

DEFAULT_REFUSALS: tuple[Refusal, ...] = (
    Refusal(
        "it would make a pathogen more dangerous (gain of function: more transmissible, virulent or lethal)",
        (
            (_re(r"\bgain[- ]of[- ]function\b"), _PATHOGEN),
            (_re(rf"\b{_RAISE}{_NEAR}{_DANGER}\b"), _PATHOGEN),
            (_re(rf"\b{_MAKE}{_NEAR}(more|highly) (transmissible|virulent|lethal|pathogenic|infectious)\b"), _PATHOGEN),
        ),
    ),
    Refusal(
        "it designs a toxin, a toxic protein or a chemical or biological weapon",
        (
            (_re(rf"\b{_MAKE}{_NEAR}{_WEAPON}\b"),),
            (_re(rf"\b{_MAKE}{_NEAR}(more|highly|maximally) (toxic|lethal|poisonous)\b"),),
        ),
        unless=_re(
            r"\b(neutrali[sz]\w*|antidotes?|antitoxins?|anti-?venoms?|detect\w*|diagnos\w*|vaccines?|inhibit\w*|antibod(y|ies))\b"
        ),
    ),
)

# Controlled-access repositories, by their accession formats (dbGaP's study submission guide; EGA's
# metadata schema page), and for resources also by host: a brief may say "not dbGaP" in words.
ACCESSIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("dbGaP", _re(r"\bph[stvdga]\d{6,8}\b")),  # phs study, pht dataset, phv variable, phd, phg, pha
    ("EGA", _re(r"\bEGA[SNXZRCPD]\d{5,}\b")),  # study, sample, experiment, analysis, run, DAC, policy, dataset
)
HOSTS: tuple[tuple[str, re.Pattern[str]], ...] = (("dbGaP", _re(r"dbgap")), ("EGA", _re(r"ega-archive\.org")))


def controlled_access(text: str, *, resource: bool = False) -> str | None:
    """The controlled-access repository a text names by accession (a resource: or by URL), if any."""
    patterns = ACCESSIONS + (HOSTS if resource else ())
    return next((name for name, pattern in patterns if pattern.search(text)), None)


def refusal_reason(texts: Iterable[str], extra: Sequence[Refusal] = ()) -> str | None:
    """Why rtool refuses these texts (the first rule that matches one of them), or None."""
    for text in texts:
        for rule in (*DEFAULT_REFUSALS, *extra):
            if rule.matches(text):
                return rule.reason
        repository = controlled_access(text)
        if repository:
            return f"it uses controlled-access data ({repository}), which rtool does not handle"
    return None


def parse_refusals(pack: str, entries: Any) -> tuple[Refusal, ...]:
    """A pack's own rules: ``[{reason, all: [regex, ...], unless?: regex}]`` (all must match)."""
    out = []
    for entry in entries or []:
        if not isinstance(entry, Mapping) or not entry.get("reason") or not entry.get("all"):
            raise ConfigError(f"pack {pack!r}: each refuse entry needs a reason and an 'all' list of patterns")
        try:
            rule = tuple(_re(str(p)) for p in entry["all"])
            unless = _re(str(entry["unless"])) if entry.get("unless") else None
        except re.error as exc:
            raise ConfigError(f"pack {pack!r}: refuse pattern does not compile: {exc}") from None
        out.append(Refusal(str(entry["reason"]), (rule,), unless))
    return tuple(out)
