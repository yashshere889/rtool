"""Does a planned resource exist, what licence does it have, how big is it?

The Hugging Face Hub API (``/api/datasets/<id>`` and ``/api/models/<id>``)
answers all three without downloading anything. URLs get a HEAD request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ...schemas.plan import ResourceChoice
from ..http import HttpError, PoliteHttp

HF_API = "https://huggingface.co/api"


@dataclass(frozen=True)
class ResourceCheck:
    exists: bool
    license: str = ""
    size_gb: float | None = None
    note: str = ""


class ResourceChecker(Protocol):
    def check(self, resource: ResourceChoice) -> ResourceCheck: ...


def _hf_size_gb(info: dict[str, Any]) -> float | None:
    used = info.get("usedStorage")
    if isinstance(used, int | float) and used > 0:
        return round(used / 1e9, 3)
    sizes = [s.get("size") for s in info.get("siblings") or [] if isinstance(s.get("size"), int)]
    return round(sum(sizes) / 1e9, 3) if sizes else None


def _hf_license(info: dict[str, Any]) -> str:
    card = info.get("cardData") or {}
    licence = card.get("license") or ""
    if not licence:
        licence = next((t.split(":", 1)[1] for t in info.get("tags") or [] if t.startswith("license:")), "")
    return licence if isinstance(licence, str) else ",".join(licence)


class HubChecker:
    """Checks resources against the Hugging Face Hub API and plain URLs."""

    def __init__(self, http: PoliteHttp, token: str | None = None) -> None:
        self.http = http
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def check(self, resource: ResourceChoice) -> ResourceCheck:
        try:
            if resource.source == "huggingface":
                kind = "models" if resource.kind == "model" else "datasets"
                info = self.http.get_json(
                    f"{HF_API}/{kind}/{resource.identifier}", params={"blobs": "true"}, headers=self.headers
                )
                gated = info.get("gated")
                note = f"gated ({gated}): needs HF_TOKEN and accepted terms" if gated else ""
                return ResourceCheck(True, _hf_license(info), _hf_size_gb(info), note)
            if resource.source == "url":
                self.http.get_text(resource.identifier, cache=False)
                return ResourceCheck(True, note="reachable")
            return ResourceCheck(True, note=f"{resource.source} resources are installed with the experiment code")
        except HttpError as exc:
            return ResourceCheck(False, note=str(exc)[:200])
