"""Polite HTTP for public scholarly APIs.

* per-host minimum spacing between requests (e.g. arXiv asks for one request
  every three seconds on a single connection);
* retries with backoff on 429 / 5xx / connection errors, honouring Retry-After;
* an on-disk response cache in ONE SQLite file (Barkla counts files against a
  quota, so no file-per-response caches);
* size-capped streaming downloads with content-type checks and checksums;
* secrets (api_key, email, ...) are stripped from anything that gets logged.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from ..core.errors import RToolError

USER_AGENT = "research-tool/0.1 (academic literature review; low request rate)"
SECRET_PARAMS = {"api_key", "apikey", "key", "email", "mailto", "token"}


class HttpError(RToolError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def redact(url: str) -> str:
    """URL with secret query parameters replaced, safe for logs and cache keys."""
    parts = urlsplit(url)
    query = [(k, "***" if k.lower() in SECRET_PARAMS else v) for k, v in parse_qsl(parts.query, keep_blank_values=True)]
    return urlunsplit(parts._replace(query=urlencode(query)))


class RateLimiter:
    """At most one request per ``min_interval_s`` (thread-safe)."""

    def __init__(
        self,
        min_interval_s: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval_s = min_interval_s
        self.clock = clock
        self.sleep = sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self._next - self.clock()
            if delay > 0:
                self.sleep(delay)
            self._next = self.clock() + self.min_interval_s


class ResponseCache:
    def __init__(self, path: Path, ttl_s: float) -> None:
        self.path = Path(path)
        self.ttl_s = ttl_s
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute(
                "create table if not exists responses (key text primary key, url text, fetched_at real, body text)"
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("pragma journal_mode=delete")  # no WAL: network filesystems
        return db

    def get(self, key: str) -> str | None:
        with self._lock, self._connect() as db:
            row = db.execute("select fetched_at, body from responses where key = ?", (key,)).fetchone()
        if row is None or time.time() - row[0] > self.ttl_s:
            return None
        return row[1]

    def put(self, key: str, url: str, body: str) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "insert or replace into responses (key, url, fetched_at, body) values (?, ?, ?, ?)",
                (key, url, time.time(), body),
            )


@dataclass(frozen=True)
class Download:
    path: Path
    sha256: str
    bytes: int
    content_type: str
    final_url: str


class PoliteHttp:
    def __init__(
        self,
        *,
        limits: Mapping[str, float] | None = None,  # host -> min seconds between requests
        default_interval_s: float = 0.2,
        cache_path: Path | None = None,
        cache_ttl_s: float = 14 * 24 * 3600,
        client: httpx.Client | None = None,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(60.0, connect=15.0), follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )
        self._limits = dict(limits or {})
        self._default = default_interval_s
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = threading.Lock()
        self.cache = ResponseCache(cache_path, cache_ttl_s) if cache_path else None
        self.max_retries = max_retries
        self._sleep = sleep
        self._clock = clock
        self.requests_made = 0

    def _limiter(self, url: str) -> RateLimiter:
        host = urlsplit(url).hostname or ""
        with self._lock:
            if host not in self._limiters:
                self._limiters[host] = RateLimiter(self._limits.get(host, self._default), self._clock, self._sleep)
            return self._limiters[host]

    def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        delay = 2.0
        problem = ""
        for attempt in range(self.max_retries + 1):
            self._limiter(url).wait()
            self.requests_made += 1
            try:
                response = self._client.request(method, url, **kwargs)
            except httpx.TransportError as exc:
                problem = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code < 400:
                    return response
                if response.status_code not in (408, 429) and response.status_code < 500:
                    raise HttpError(
                        f"HTTP {response.status_code} for {redact(str(response.url))}", response.status_code
                    )
                problem = f"HTTP {response.status_code}"
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = max(delay, min(float(retry_after), 300.0))
            if attempt < self.max_retries:
                self._sleep(delay)
                delay = min(delay * 2, 120.0)
        raise HttpError(f"giving up on {redact(url)} after {self.max_retries + 1} attempts ({problem})")

    def get_text(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        cache: bool = True,
    ) -> str:
        full = str(httpx.URL(url, params={k: v for k, v in (params or {}).items() if v is not None}))
        key = hashlib.sha256(redact(full).encode()).hexdigest()
        if cache and self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        body = self._send("GET", full, headers=dict(headers or {})).text
        if cache and self.cache is not None:
            self.cache.put(key, redact(full), body)
        return body

    def get_json(self, url: str, **kwargs: Any) -> Any:
        text = self.get_text(url, **kwargs)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise HttpError(f"invalid JSON from {redact(url)}: {text[:200]!r}") from None

    def download(
        self,
        url: str,
        dest: Path,
        *,
        max_bytes: int = 60 * 1024 * 1024,
        accept: Iterable[str] = ("application/pdf", "text/html", "application/xhtml+xml"),
    ) -> Download:
        """Stream ``url`` to ``dest`` (atomically); refuse oversized or unexpected content."""
        accepted = tuple(accept)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.part")
        self._limiter(url).wait()
        self.requests_made += 1
        digest = hashlib.sha256()
        size = 0
        try:
            with self._client.stream("GET", url) as response:
                if response.status_code >= 400:
                    raise HttpError(f"HTTP {response.status_code} downloading {redact(url)}", response.status_code)
                content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if accepted and content_type not in accepted:
                    raise HttpError(f"unexpected content type {content_type!r} from {redact(url)}")
                with open(tmp, "wb") as out:
                    for block in response.iter_bytes():
                        size += len(block)
                        if size > max_bytes:
                            raise HttpError(f"{redact(url)} is larger than {max_bytes // (1024 * 1024)} MB")
                        digest.update(block)
                        out.write(block)
                final_url = str(response.url)
            os.replace(tmp, dest)
        except httpx.TransportError as exc:
            raise HttpError(f"download failed for {redact(url)}: {exc}") from None
        finally:
            tmp.unlink(missing_ok=True)
        return Download(dest, "sha256:" + digest.hexdigest(), size, content_type, redact(final_url))

    def close(self) -> None:
        self._client.close()
