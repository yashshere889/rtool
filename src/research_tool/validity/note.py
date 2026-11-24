"""What a report may say about how far its field can be trusted, from the field's newest validity card.

Cards are written by `rtool validity score` into ``<runs root>/.validity-cards``. A field with no card has never
been measured, and a report of it must not suggest otherwise (docs/roadmap.md, "Validity levels"). The writer puts
this sentence into every report it writes, and the web pages show the same text.
"""

from __future__ import annotations

from pathlib import Path

from .card import Card

CARDS = ".validity-cards"
NEVER = "A new finding is never claimed as trustworthy: that level needs replication by people."


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


def statement(pack: str, card: Card | None) -> str:
    """The sentence a report of ``pack`` carries: nothing measured, or what the card says and its limit."""
    if card is None or not card.runs:
        return (
            f"No validity card exists for the {pack} field on this server, so no level is claimed: the results show "
            "what this run did, not that rtool is known to get such questions right."
        )
    model, code = card.provenance.get("model", "?"), card.provenance.get("code", "?")
    parts = ", ".join(f"{kind} {r.k}/{r.n}" for kind, r in card.headline.items())
    fc = card.false_claims
    return (
        f"Measured on {card.runs} scored benchmark runs of the {pack} field (model {model}, code {code}); runs that "
        f"reached their case's top level: {parts or 'none finished'}"
        + (f"; false claims on null cases {fc.k}/{fc.n}" if fc.n else "")
        + f". {NEVER}"
    )


def statement_for(runs_root: Path, pack: str) -> str:
    return statement(pack, latest_cards(runs_root).get(pack))
