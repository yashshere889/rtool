"""How far a field's results can be trusted: its validity card, and what a run's report may say about it.

A card is written by `rtool validity score` into ``<runs root>/.validity-cards`` and says how often rtool reached
the right answer on benchmark cases of that field, with model and code named. A field with no card has never been
measured, and a report of it must not suggest otherwise (docs/roadmap.md, "Validity levels"). Read-only.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ...validity.card import Card
from ..deps import get_run, settings_of

router = APIRouter(tags=["validity"])

CARDS = ".validity-cards"
NEVER = "A new finding is never claimed as trustworthy: that level needs replication by people."


class KindRate(BaseModel):
    kind: str
    k: int
    n: int
    low: float
    high: float


class ValidityNote(BaseModel):
    pack: str
    measured: bool
    text: str  # what a reader may take from this field's card, in a sentence or two
    runs: int
    top_level: list[KindRate]
    false_claims: KindRate | None
    model: str
    code: str
    created_at: str


def latest_cards(runs_root: Path) -> dict[str, Card]:
    """The newest card of each pack under ``runs_root``; unreadable files are skipped."""
    found: dict[str, Card] = {}
    folder = Path(runs_root).expanduser() / CARDS
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            card = Card.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if card.pack not in found or card.created_at > found[card.pack].created_at:
            found[card.pack] = card
    return found


def note_for(pack: str, card: Card | None) -> ValidityNote:
    if card is None or not card.runs:
        text = (
            f"No validity card exists for the {pack} field on this server, so no level is claimed: the results show "
            "what this run did, not that rtool is known to get such questions right."
        )
        return ValidityNote(
            pack=pack,
            measured=False,
            text=text,
            runs=0,
            top_level=[],
            false_claims=None,
            model="",
            code="",
            created_at="",
        )
    top = [KindRate(kind=k, k=r.k, n=r.n, low=r.low, high=r.high) for k, r in card.headline.items()]
    fc = card.false_claims
    model, code = str(card.provenance.get("model", "?")), str(card.provenance.get("code", "?"))
    parts = ", ".join(f"{t.kind} {t.k}/{t.n}" for t in top)
    text = (
        f"Measured on {card.runs} scored benchmark runs of the {pack} field (model {model}, code {code}); runs that "
        f"reached their case's top level: {parts or 'none finished'}"
        + (f"; false claims on null cases {fc.k}/{fc.n}" if fc.n else "")
        + f". {NEVER}"
    )
    return ValidityNote(
        pack=pack,
        measured=True,
        text=text,
        runs=card.runs,
        top_level=top,
        false_claims=KindRate(kind="null", k=fc.k, n=fc.n, low=fc.low, high=fc.high) if fc.n else None,
        model=model,
        code=code,
        created_at=card.created_at,
    )


@router.get("/validity", response_model=list[Card])
def cards(request: Request) -> list[Card]:
    """The newest validity card of each field, as `rtool validity score` wrote it."""
    return sorted(latest_cards(settings_of(request).runs_root).values(), key=lambda c: c.pack)


@router.get("/runs/{project}/{run_id}/validity", response_model=ValidityNote)
def run_validity(project: str, run_id: str, request: Request) -> ValidityNote:
    """What this run's field has been measured at (nothing, if it has no card)."""
    run = get_run(request, project, run_id)
    return note_for(run.spec.pack, latest_cards(settings_of(request).runs_root).get(run.spec.pack))
