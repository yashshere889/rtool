"""Everything the Literature agent talks to, built from config and environment.

Tests (and other sites) inject their own toolkit through
``Services.tools["literature_toolkit"]`` — a callable ``(config, cache_path) -> LiteratureToolkit``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.config import KnowledgeConfig
from ...knowledge.vectors import Embedder, make_embedder
from ...tools.documents.convert import pymupdf_to_markdown
from ...tools.http import PoliteHttp
from ...tools.scholarly.sources import ArxivSource, OpenAlexSource, SemanticScholarSource, UnpaywallResolver

# Seconds between requests per host, from each service's published terms.
HOST_LIMITS = {
    "export.arxiv.org": 3.1,  # arXiv API: one request every three seconds, single connection
    "arxiv.org": 3.1,  # arXiv downloads: same courtesy
    "api.semanticscholar.org": 1.1,  # about one request per second with a free key
    "api.openalex.org": 0.12,
    "api.unpaywall.org": 0.2,
}


@dataclass
class LiteratureToolkit:
    sources: dict[str, Any]  # name -> source with .search(); optional neighbour methods
    unpaywall: UnpaywallResolver | None
    http: PoliteHttp  # used for downloads
    pdf_converter: Callable[[Path], str]
    embedder: Embedder

    def close(self) -> None:
        self.http.close()


def build_toolkit(
    config: Any, cache_path: Path, knowledge: KnowledgeConfig, environ: Mapping[str, str] | None = None
) -> LiteratureToolkit:
    environ = os.environ if environ is None else environ
    limits = dict(HOST_LIMITS)
    if not environ.get("S2_API_KEY"):
        limits["api.semanticscholar.org"] = 3.5  # keyless callers share one throttled pool
    http = PoliteHttp(limits=limits, cache_path=cache_path)
    sources: dict[str, Any] = {}
    if "openalex" in config.sources:
        sources["openalex"] = OpenAlexSource(http, api_key=environ.get("OPENALEX_API_KEY"))
    if "semantic_scholar" in config.sources:
        sources["semantic_scholar"] = SemanticScholarSource(http, api_key=environ.get("S2_API_KEY"))
    if "arxiv" in config.sources:
        sources["arxiv"] = ArxivSource(http)
    email = environ.get("UNPAYWALL_EMAIL")
    return LiteratureToolkit(
        sources=sources,
        unpaywall=UnpaywallResolver(http, email) if email else None,
        http=http,
        pdf_converter=pymupdf_to_markdown,
        embedder=make_embedder(knowledge.embedder, knowledge.embedder_model, cache_dir=knowledge.cache_dir),
    )
