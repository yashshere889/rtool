"""Check that a quote the model says it copied from a paper is really in the paper.

This is the main guard against a 12B model inventing evidence: every claim the
Literature agent keeps must carry a quote that matches the paper text. Matching
is tolerant of PDF artefacts (hyphenation, ligatures, spacing, punctuation) but
not of paraphrase.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_NON_WORD = re.compile(r"[^0-9a-z]+")
MIN_QUOTE_CHARS = 15
DEFAULT_THRESHOLD = 0.85


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # also folds ligatures like "ﬁ"
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return " ".join(_NON_WORD.sub(" ", text.lower()).split())


def quote_support(quote: str, source: str, *, normalized_source: str | None = None) -> float:
    """Fraction (0..1) of the quote found, in order, in the best-matching stretch of the source."""
    q = normalize(quote)
    if len(q) < MIN_QUOTE_CHARS:
        return 0.0
    s = normalized_source if normalized_source is not None else normalize(source)
    if q in s:
        return 1.0
    anchors = sorted({w for w in q.split() if len(w) > 3}, key=len, reverse=True)[:4]
    best = 0.0
    for anchor in anchors:
        start = 0
        for _ in range(25):  # bounded number of anchor occurrences
            pos = s.find(anchor, start)
            if pos < 0:
                break
            window = s[max(0, pos - len(q)) : pos + len(q)]
            matcher = difflib.SequenceMatcher(None, q, window, autojunk=False)
            best = max(best, sum(block.size for block in matcher.get_matching_blocks()) / len(q))
            if best >= 0.98:
                return round(best, 3)
            start = pos + 1
    return round(best, 3)


def is_supported(quote: str, source: str, threshold: float = DEFAULT_THRESHOLD) -> bool:
    return quote_support(quote, source) >= threshold
