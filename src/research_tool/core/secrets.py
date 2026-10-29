"""API keys from ``~/.config/rtool/secrets.env`` (never printed, never stored in runs).

Format: ``KEY=VALUE`` per line, ``#`` comments. Keys used by the tool:

    OPENALEX_API_KEY   free key from openalex.org (their docs require one)
    S2_API_KEY         free Semantic Scholar API key (higher, dedicated rate limit)
    UNPAYWALL_EMAIL    contact email Unpaywall asks for on every request
    HF_TOKEN           only for gated Hugging Face datasets/models

The file must be private (``chmod 600``). Variables already set in the
environment win, so a job can override them.

Secrets stay in the rtool process that loaded them: environments handed to
anything else (experiment code written by the model, the vLLM server, Slurm
jobs via ``sbatch``) go through :func:`without_secrets`. Episode jobs load the
file again themselves.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from .errors import RToolError

DEFAULT_PATH = "~/.config/rtool/secrets.env"
SECRET_NAMES = frozenset({"OPENALEX_API_KEY", "S2_API_KEY", "UNPAYWALL_EMAIL", "HF_TOKEN", "RTOOL_SECRETS"})
_loaded: set[str] = set()  # names read from a secrets file by this process


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
    _loaded.update(values)
    return sorted(values)


def without_secrets(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """A copy of ``environ`` without API keys, for processes rtool starts but does not trust with them."""
    environ = os.environ if environ is None else environ
    hidden = SECRET_NAMES | _loaded
    return {k: v for k, v in environ.items() if k not in hidden}


def secrets_status(environ: Mapping[str, str] | None = None) -> dict[str, bool]:
    """Which of the known keys are set (never their values), e.g. for a settings page."""
    environ = os.environ if environ is None else environ
    return {name: bool(environ.get(name)) for name in sorted(SECRET_NAMES - {"RTOOL_SECRETS"})}
