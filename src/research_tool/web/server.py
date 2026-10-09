"""Starting the UI server (``rtool ui``)."""

from __future__ import annotations

import os
import secrets
import webbrowser
from pathlib import Path

import uvicorn

from .app import create_app
from .settings import UISettings


def new_token() -> str:
    return secrets.token_urlsafe(24)


def write_private(path: Path, text: str) -> None:
    """Write a file only its owner can read (the token file on a shared cluster)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(path, 0o600)


def serve(settings: UISettings, *, host: str, port: int, open_browser: bool = False) -> None:
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/#token={settings.token}"
    print(f"Research Tool UI ({settings.mode} mode, runs in {settings.runs_root})")
    print(f"open: {url}")
    print("stop with Ctrl-C")
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(create_app(settings), host=host, port=port, log_level="warning", proxy_headers=False)
