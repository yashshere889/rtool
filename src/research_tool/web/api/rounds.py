"""What changed since the previous version of a stage's output, and why: the comments the stage acted on.

A version records the attempt that made it (``producer.attempt``); a comment records the attempt that used
it (``consumed_by_attempt``). A request for changes starts a new attempt, so the two pair up. An attempt
whose output is identical to the latest version makes no new version (the store keeps the old one), so
comments used after the latest version was made mean the stage ran again and came back unchanged.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ...core.artifacts import ArtifactEnvelope, ArtifactStore
from ...core.run import RunHandle
from ...core.state import FeedbackItem, FeedbackKind, StateStore

# texts the engine writes when a decision came without words (core/engine.py); not the researcher's own
_PLACEHOLDERS = {"see the researcher's choices", "rejected without comment"}


class RoundFeedback(BaseModel):
    by: Literal["you", "stage"]  # the researcher, or another stage asking for a revision
    from_stage: str | None = None
    text: str = ""  # empty: no comment, only choices
    kept: list[str] = Field(default_factory=list)  # titles of papers marked Keep
    dropped: list[str] = Field(default_factory=list)  # titles of papers marked Drop


class Round(BaseModel):
    """The version shown against the one before it."""

    previous_version: int
    previous_at: str
    feedback: list[RoundFeedback]  # what the stage was asked to change (empty: none recorded)
    inputs_changed: list[str]  # earlier outputs that were newer for this version
    unchanged: bool = False  # the stage ran again on ``feedback`` and produced this same version


def previous_round(
    run: RunHandle, store: ArtifactStore, envelope: ArtifactEnvelope, titles: dict[str, str] | None = None
) -> tuple[Round, ArtifactEnvelope] | None:
    """``envelope`` against the version before it, and that version (None when there is nothing to compare)."""
    state = StateStore(run.state_dir).load_stage(envelope.producer.stage)
    feedback = state.feedback if state else []
    name = (titles or {}).get

    def view(items: list[FeedbackItem]) -> list[RoundFeedback]:
        return [
            RoundFeedback(
                by="stage" if f.kind is FeedbackKind.CHANGE_REQUEST else "you",
                from_stage=f.from_stage,
                text="" if f.text in _PLACEHOLDERS else f.text,
                kept=[name(i, i) for i in f.data.get("include", [])],
                dropped=[name(i, i) for i in f.data.get("exclude", [])],
            )
            for f in items
        ]

    versions = sorted(store.versions(envelope.name))
    attempt = envelope.producer.attempt
    later = [f for f in feedback if f.consumed_by_attempt is not None and f.consumed_by_attempt > attempt]
    if later and envelope.version == versions[-1]:
        latest = max(f.consumed_by_attempt or 0 for f in later)
        same = Round(
            previous_version=envelope.version,
            previous_at=envelope.created_at,
            feedback=view([f for f in later if f.consumed_by_attempt == latest]),
            inputs_changed=[],
            unchanged=True,
        )
        return same, envelope
    earlier = [v for v in versions if v < envelope.version]
    if not earlier:
        return None
    previous = store.envelope(envelope.name, earlier[-1])
    changed = sorted(k for k, v in envelope.inputs.items() if previous.inputs.get(k) != v)
    used = [f for f in feedback if f.consumed_by_attempt == attempt]
    return Round(
        previous_version=previous.version, previous_at=previous.created_at, feedback=view(used), inputs_changed=changed
    ), previous
