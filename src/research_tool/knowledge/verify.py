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


_WORD = re.compile(r"[0-9A-Za-z]+")
_CLOSING = ".,;:!?)]\"'\u2019\u201d"


def _with_closing(quote: str, text: str, end: int) -> int:
    """Extend a match over punctuation that ends the quote (e.g. its full stop), if the text has it too."""
    stripped = quote.rstrip()
    closing = stripped[len(stripped.rstrip(_CLOSING)) :]
    return end + len(closing) if closing and text.startswith(closing, end) else end


def locate_quote(quote: str, text: str, threshold: float = DEFAULT_THRESHOLD) -> tuple[int, int, float] | None:
    """Where ``quote`` is in ``text``: (start, end, score) as character offsets into ``text``, or None.

    Matches word by word, so the offsets point into the original text despite the
    differences ``normalize`` forgives (case, punctuation, spacing, line breaks).
    Used to highlight a verified claim in the paper it came from.
    """
    q = [w.lower() for w in _WORD.findall(unicodedata.normalize("NFKC", quote))]
    if not q:
        return None
    words = [(m.group().lower(), m.start(), m.end()) for m in _WORD.finditer(text)]
    tokens = [w for w, _, _ in words]
    n = len(q)
    for i in range(len(tokens) - n + 1):  # exact word sequence: the common case
        if tokens[i : i + n] == q:
            return words[i][1], _with_closing(quote, text, words[i + n - 1][2]), 1.0
    anchors = sorted({w for w in q if len(w) > 3}, key=len, reverse=True)[:4]
    best: tuple[int, int, float] | None = None
    seen: set[int] = set()
    for anchor in anchors:
        for pos in [i for i, t in enumerate(tokens) if t == anchor][:25]:
            start = max(0, pos - n)
            if start in seen:
                continue
            seen.add(start)
            window = tokens[start : pos + n]
            blocks = [
                b for b in difflib.SequenceMatcher(None, q, window, autojunk=False).get_matching_blocks() if b.size
            ]
            if not blocks:
                continue
            score = sum(b.size for b in blocks) / n
            if best is None or score > best[2]:
                first, last = start + blocks[0].b, start + blocks[-1].b + blocks[-1].size - 1
                best = (words[first][1], _with_closing(quote, text, words[last][2]), round(score, 3))
    return best if best is not None and best[2] >= threshold else None
