"""Split paper text into sections, separate the reference list, and chunk for retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# "1 Introduction", "2. Related Work", "IV. RESULTS", "A Appendix" (common in PDF conversions)
_NUMBERED_HEADING = re.compile(
    r"^(?:\d{1,2}(?:\.\d{1,2})*\.?|[IVX]{1,5}\.|[A-H]\.?)\s+([A-Z][A-Za-z][\w\s,:&()-]{1,70})$"
)
_BARE_HEADING = re.compile(
    r"^(abstract|introduction|background|related work|methods?|methodology|materials and methods|experiments?|"
    r"results|discussion|conclusions?|limitations|future work|acknowledge?ments|references|bibliography)\s*$",
    re.I,
)
_REFERENCE_TITLES = re.compile(r"^(references|bibliography|works cited|literature cited)\b", re.I)
_TAIL_TITLES = re.compile(r"^(acknowledge?ments|funding|author contributions|appendix|supplementary)", re.I)
_PRIORITY = re.compile(
    r"abstract|introduction|method|approach|experiment|result|discussion|conclusion|limitation", re.I
)


@dataclass(frozen=True)
class Section:
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    seq: int
    section: str
    text: str


def _heading(line: str) -> str | None:
    stripped = line.strip().strip("*_").strip()
    if not stripped or len(stripped) > 90:
        return None
    md = _MD_HEADING.match(stripped)
    if md:
        return md.group(2).strip("*_ ")
    numbered = _NUMBERED_HEADING.match(stripped)
    if numbered and not stripped.endswith("."):
        return numbered.group(1).strip()
    if _BARE_HEADING.match(stripped):
        return stripped
    return None


def split_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    title, buffer = "front matter", []
    for line in text.splitlines():
        heading = _heading(line)
        if heading is not None:
            if "".join(buffer).strip():
                sections.append(Section(title, "\n".join(buffer).strip()))
            title, buffer = heading, []
        else:
            buffer.append(line)
    if "".join(buffer).strip():
        sections.append(Section(title, "\n".join(buffer).strip()))
    return sections


def main_text(sections: list[Section]) -> tuple[list[Section], list[Section]]:
    """(body, references): everything from the reference list on is set aside."""
    for i, section in enumerate(sections):
        if _REFERENCE_TITLES.match(section.title):
            return sections[:i], sections[i:]
    return sections, []


def budgeted_text(sections: list[Section], max_chars: int) -> str:
    """Paper text for one LLM call: core sections first, tail material last, cut at ``max_chars``."""
    body = [s for s in sections if not _TAIL_TITLES.match(s.title)]
    ordered = [s for s in body if _PRIORITY.search(s.title)] + [s for s in body if not _PRIORITY.search(s.title)]
    if not ordered:
        ordered = body
    position = {id(s): i for i, s in enumerate(body)}
    chosen, used = [], 0
    for section in ordered:
        block = f"## {section.title}\n{section.text}\n"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 500:
                chosen.append((position[id(section)], block[:remaining]))
            break
        chosen.append((position[id(section)], block))
        used += len(block)
    return "\n".join(block for _, block in sorted(chosen))  # keep the paper's own order


def chunk_sections(sections: list[Section], max_chars: int = 4000, overlap_chars: int = 300) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        text = section.text
        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            if end < len(text):  # prefer to break at a paragraph or sentence boundary
                window = text[start:end]
                cut = max(window.rfind("\n\n"), window.rfind(". "))
                if cut > max_chars // 2:
                    end = start + cut + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(seq=len(chunks), section=section.title, text=piece))
            if end >= len(text):
                break
            start = max(end - overlap_chars, start + 1)
    return chunks
