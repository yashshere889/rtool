"""HTML -> Markdown-ish text (standard library only), and PDF -> Markdown (Barkla only).

arXiv publishes HTML versions of many recent papers; they convert to clean text
with headings and no extra dependencies. PDFs need PyMuPDF (``pymupdf4llm``,
installed in the Barkla environment via the ``[barkla]`` extra).
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol

from ...core.errors import RToolError

_SKIP = {"script", "style", "nav", "footer", "header", "noscript", "svg", "button", "form"}
_BLOCK = {"p", "div", "section", "article", "li", "tr", "br", "figcaption", "blockquote", "table", "dd", "dt"}
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_BLANKS = re.compile(r"\n{3,}")


class ConversionError(RToolError):
    """A document could not be turned into text."""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self._heading: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "math":
            alt = dict(attrs).get("alttext")
            if alt:
                self.parts.append(f" {alt} ")
            self._skip_depth += 1  # skip the MathML tree itself
            return
        if self._heading is not None and (tag in _HEADINGS or tag in _BLOCK):
            # A block starting inside a heading means the heading was never closed (malformed HTML):
            # end it here, as browsers do, so it doesn't swallow the rest of the document.
            self._heading = None
            self.parts.append("\n\n")
        if tag in _HEADINGS:
            self._heading = _HEADINGS[tag]
            self.parts.append("\n\n" + "#" * self._heading + " ")
        elif tag in _BLOCK:
            self.parts.append("\n")
        elif tag == "td":
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP or tag == "math":
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in _HEADINGS:
            self._heading = None
            self.parts.append("\n\n")
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data if self._heading is None else data.replace("\n", " "))


def html_to_text(markup: str) -> str:
    parser = _TextExtractor()
    parser.feed(markup)
    parser.close()
    text = html.unescape("".join(parser.parts))
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return _BLANKS.sub("\n\n", "\n".join(lines)).strip() + "\n"


class PdfConverter(Protocol):
    def __call__(self, path: Path) -> str: ...


def pymupdf_to_markdown(path: Path) -> str:
    try:
        import pymupdf4llm  # type: ignore[import-not-found]
    except ImportError:
        raise ConversionError("PDF parsing needs pymupdf4llm (installed on Barkla via the [barkla] extra)") from None
    return str(pymupdf4llm.to_markdown(str(path), show_progress=False))


def document_to_text(path: Path, *, pdf_converter: PdfConverter = pymupdf_to_markdown) -> str:
    """Text of a downloaded paper, chosen by file contents rather than extension."""
    data = Path(path).read_bytes()
    if data[:5] == b"%PDF-":
        return pdf_converter(Path(path))
    head = data[:2048].lower()
    if b"<html" in head or b"<!doctype html" in head:
        return html_to_text(data.decode("utf-8", errors="replace"))
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ConversionError(f"{path} is neither PDF, HTML nor UTF-8 text") from None
