"""API keys from ``~/.config/rtool/secrets.env`` (never printed, never stored in runs).

Format: ``KEY=VALUE`` per line, ``#`` comments. Keys used by the tool:

    OPENALEX_API_KEY   free key from openalex.org (their docs require one)
    S2_API_KEY         free Semantic Scholar API key (higher, dedicated rate limit)
    UNPAYWALL_EMAIL    contact email Unpaywall asks for on every request
    HF_TOKEN           only for gated Hugging Face datasets/models

The file must be private (``chmod 600``). Variables already set in the
environment win, so a job can override them.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

from .errors import RToolError

DEFAULT_PATH = "~/.config/rtool/secrets.env"


class SecretsError(RToolError):
    """The secrets file is unreadable or not private."""


def parse_env_file(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def secrets_path(environ: MutableMapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    return Path(environ.get("RTOOL_SECRETS", DEFAULT_PATH)).expanduser()


def load_secrets(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Load the secrets file into ``environ`` (without overriding). Returns the key names found."""
    environ = os.environ if environ is None else environ
    path = secrets_path(environ)
    if not path.is_file():
        return []
    if path.stat().st_mode & 0o077:
        raise SecretsError(f"{path} is readable by other users; run: chmod 600 {path}")
    values = parse_env_file(path.read_text(encoding="utf-8"))
    for key, value in values.items():
        environ.setdefault(key, value)
    return sorted(values)
