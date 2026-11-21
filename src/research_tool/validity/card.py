"""The validity card: how often a pack's runs reached the right answer, with honest intervals.

Rates are Wilson 95 % intervals, which stay sensible at the run counts that can be afforded: 0 false
claims in 10 null runs still allows a true rate of about 28 %. Cases whose expected outcome nobody
has confirmed yet are reported apart, never in the headline (docs/stage0_design.md, rule 2).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.fs import now_iso, read_json
from ..core.run import RunHandle
from ..core.stats import stats_for_run_dir
from .runs import MARKER
from .score import LEVELS, RunScore, score_run

LIMITS = [
    "Cases start at the plan stage with a fixed hypothesis, conditions and primary metric: this measures "
    "whether rtool carries out a clear question correctly, not literature search, hypothesis quality or "
    "experiment design from an open question (open cases are scored for finishing and consistent code only).",
    "Every number is for the model and code named under provenance; another model or code means new runs.",
    "L4 (a new finding is trustworthy) is not measured: that needs replication by people.",
]


class Rate(BaseModel):
    k: int
    n: int
    low: float  # Wilson 95 % interval
    high: float

    @property
    def share(self) -> float:
        return self.k / self.n if self.n else 0.0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(k: int, n: int) -> Rate:
    low, high = wilson(k, n)
    return Rate(k=k, n=n, low=round(low, 4), high=round(high, 4))


class Card(BaseModel):
    pack: str
    created_at: str = Field(default_factory=now_iso)
    provenance: dict[str, Any] = Field(default_factory=dict)  # code, commit, dirty, model
    runs: int = 0  # scored runs
    unfinished: list[str] = Field(default_factory=list)  # still going: not scored
    headline: dict[str, Rate] = Field(default_factory=dict)  # kind -> runs that reached their top level
    levels: dict[str, dict[str, int]] = Field(default_factory=dict)  # kind -> {"none", "L0".."L3": runs}
    false_claims: Rate = Field(default_factory=lambda: rate(0, 0))  # null runs whose report claims support
    variants: dict[str, dict[str, Rate]] = Field(default_factory=dict)  # case -> variant -> top-level rate
    unconfirmed: dict[str, Rate] = Field(default_factory=dict)  # case -> top-level rate, outside the headline
    categories: dict[str, list[str]] = Field(default_factory=dict)  # failure category -> run ids
    cost: dict[str, float] = Field(default_factory=dict)  # totals over scored runs
    limits: list[str] = Field(default_factory=lambda: list(LIMITS))


def score_runs(runs_root: Path, pack: str, version: str | None = None) -> list[tuple[RunHandle, RunScore]]:
    """Every validity run of ``pack`` (of code versions starting with ``version``, if given), scored."""
    project = Path(runs_root).expanduser() / f"validity-{pack}"
    found = []
    for folder in sorted(p for p in project.iterdir() if (p / MARKER).is_file()) if project.is_dir() else []:
        if version and not str(read_json(folder / MARKER).get("commit", "")).startswith(version):
            continue
        run = RunHandle(folder)
        found.append((run, score_run(run)))
    return found


def run_cost(run: RunHandle) -> dict[str, float]:
    stats = stats_for_run_dir(run.root)
    return {
        "llm_calls": stats.llm.calls,
        "prompt_tokens": stats.llm.prompt_tokens,
        "completion_tokens": stats.llm.completion_tokens,
        "wall_hours": stats.wall.total_s / 3600,
        "gpu_episode_hours": stats.wall.gpu_run_s / 3600,
    }


def build_card(
    pack: str,
    scored: list[tuple[RunHandle | None, RunScore]],
    *,
    provenance: dict[str, Any] | None = None,
    cost_of: Callable[[RunHandle], dict[str, float]] = run_cost,
) -> Card:
    card = Card(pack=pack, provenance=dict(provenance or {}))
    done = [(run, s) for run, s in scored if s.finished]
    card.unfinished = [s.run_id for _, s in scored if not s.finished]
    card.runs = len(done)

    def top(s: RunScore) -> bool:
        return s.level == s.max_level

    headline: dict[str, list[bool]] = defaultdict(list)
    unconfirmed: dict[str, list[bool]] = defaultdict(list)
    variants: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    levels: dict[str, Counter[str]] = defaultdict(Counter)
    categories: dict[str, list[str]] = defaultdict(list)
    nulls = claims = 0
    for _, s in done:
        (headline[s.kind] if s.confirmed else unconfirmed[s.case]).append(top(s))
        variants[s.case][s.variant].append(top(s))
        levels[s.kind]["none" if s.level < 0 else LEVELS[s.level].split()[0]] += 1
        if s.category:
            categories[s.category].append(s.run_id)
        if s.kind == "null":
            nulls += 1
            claims += s.false_claim
    card.headline = {kind: rate(sum(v), len(v)) for kind, v in sorted(headline.items())}
    card.unconfirmed = {case: rate(sum(v), len(v)) for case, v in sorted(unconfirmed.items())}
    card.variants = {
        case: {name: rate(sum(v), len(v)) for name, v in sorted(by.items())}
        for case, by in sorted(variants.items())
        if len(by) > 1
    }
    card.levels = {kind: dict(sorted(c.items())) for kind, c in sorted(levels.items())}
    card.categories = dict(sorted(categories.items()))
    card.false_claims = rate(claims, nulls)
    totals: Counter[str] = Counter()
    for run, _ in done:
        if run is None:
            continue
        try:
            totals.update(cost_of(run))
        except Exception:  # noqa: BLE001 - a run without a readable event log still counts, just not its cost
            totals["runs_without_cost"] += 1
    card.cost = {key: round(value, 3) for key, value in sorted(totals.items())}
    return card


def _rate_text(r: Rate) -> str:
    return f"{r.k} / {r.n} ({100 * r.share:.0f} %, 95 % interval {100 * r.low:.0f}–{100 * r.high:.0f} %)"


def render_markdown(card: Card) -> str:
    p = card.provenance
    lines = [
        f"# Validity card: {card.pack}",
        "",
        f"{card.created_at}. Code `{str(p.get('code', '?'))[:12]}`"
        + (f", commit `{str(p['commit'])[:10]}`" if p.get("commit") else ", no git commit")
        + (" **with uncommitted changes**" if p.get("dirty") else "")
        + (f"; model `{p['model']}`" if p.get("model") else "")
        + ".",
        "",
        "## Headline (confirmed cases)",
        "",
        "Runs that reached their case's top level (L3; L1 for open cases):",
        "",
    ]
    lines += [f"- **{kind}**: {_rate_text(r)}" for kind, r in card.headline.items()] or ["- no scored runs yet"]
    lines += ["", f"**False claims** on null cases (the report says supported): {_rate_text(card.false_claims)}", ""]
    if card.levels:
        lines += ["## Highest level reached", "", "| kind | none | L0 | L1 | L2 | L3 |", "|---|---|---|---|---|---|"]
        for kind, counts in card.levels.items():
            cells = " | ".join(str(counts.get(k, 0)) for k in ("none", "L0", "L1", "L2", "L3"))
            lines.append(f"| {kind} | {cells} |")
        lines.append("")
    if card.variants:
        lines += ["## By variant", ""]
        for case, by in card.variants.items():
            lines += [f"- {case}: " + "; ".join(f"{name} {r.k}/{r.n}" for name, r in by.items())]
        lines.append("")
    if card.unconfirmed:
        lines += ["## Not in the headline: expected outcome not confirmed yet", ""]
        lines += [f"- {case}: {_rate_text(r)}" for case, r in card.unconfirmed.items()]
        lines.append("")
    if card.categories:
        lines += ["## Failures", ""]
        lines += [f"- `{category}`: {len(ids)} ({', '.join(ids[:5])})" for category, ids in card.categories.items()]
        lines.append("")
    if card.unfinished:
        lines += [f"Still running, not scored: {', '.join(card.unfinished)}", ""]
    if card.cost:
        lines += ["## Cost", ""] + [f"- {key}: {value:g}" for key, value in card.cost.items()] + [""]
    lines += ["## What this card does not measure", ""] + [f"- {limit}" for limit in card.limits] + [""]
    return "\n".join(lines)
