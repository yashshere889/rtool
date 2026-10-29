"""Helpers for testing the web API (needs the ``web`` extra)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from ..service import KickResult
from ..web.app import create_app
from ..web.runner import WorkerState
from ..web.settings import UISettings

TOKEN = "test-token-0123456789abcdef"
BASE = "http://127.0.0.1:8765"
WRITE = {"x-requested-with": "rtool"}


class StubRunner:
    """Records wake-ups and advances nothing, so queued actions stay queued."""

    def __init__(self) -> None:
        self.woken: list[str] = []

    def wake(self, run: Any, reason: str, *, needs_llm: bool = False) -> KickResult:
        self.woken.append(reason)
        return KickResult(submitted=False, detail="stub")

    def state(self, run: Any) -> WorkerState:
        return WorkerState()


def web_client(runs_root: Path, runner: Any = None, *, signed_in: bool = True) -> TestClient:
    """A test client for a local-mode server over ``runs_root``, signed in unless asked otherwise."""
    app = create_app(UISettings(runs_root=runs_root, token=TOKEN, mode="local"), runner=runner or StubRunner())
    client = TestClient(app, base_url=BASE)
    if signed_in:
        assert client.post("/api/session", json={"token": TOKEN}, headers=WRITE).status_code == 204
    return client
