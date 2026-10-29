"""Who may talk to the server: the person holding the token, through a loopback address.

The server runs on shared machines (login and compute nodes), where any user
can open a connection to it, and it is reached from a browser that also visits
other sites. So:

* every API call needs the session cookie (set by trading the token printed at
  start-up) or ``Authorization: Bearer <token>``;
* the ``Host`` header must be a loopback name (127.0.0.1, localhost, ::1), which
  blocks DNS-rebinding pages;
* state-changing calls need ``X-Requested-With: rtool`` (a header other sites
  cannot send without a CORS preflight, which is never granted) and, if the
  browser sends one, a loopback ``Origin``;
* responses carry headers that stop sniffing, framing by other sites and
  referrer leaks;
* request bodies are capped (paper uploads 64 MB, everything else 8 MB);
* the session cookie is named after the port, because browsers send a host's
  cookies to every port: two servers on one machine must not sign each
  other out.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Awaitable, Callable, MutableMapping
from http.cookies import SimpleCookie
from typing import Any
from urllib.parse import urlsplit

SESSION_COOKIE = "rtool_session"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
PUBLIC_PATHS = frozenset({"/api/health", "/api/session"})
MUTATING = frozenset({"POST", "PUT", "PATCH", "DELETE"})
MAX_BODY = 8 * 1024 * 1024
MAX_UPLOAD_BODY = 64 * 1024 * 1024  # PUT …/literature/inbox (the route itself allows 60 MB)
SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"x-frame-options", b"SAMEORIGIN"),
    (b"cross-origin-opener-policy", b"same-origin"),
]

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def session_value(token: str) -> str:
    """What the session cookie holds: derived from the token, so the token itself never sits in a cookie."""
    return hmac.new(token.encode(), b"rtool-ui-session", hashlib.sha256).hexdigest()


def hostname(netloc: str) -> str:
    """'127.0.0.1:8765' -> '127.0.0.1', '[::1]:80' -> '::1', 'localhost' -> 'localhost'."""
    netloc = netloc.strip().lower()
    if netloc.startswith("["):
        return netloc[1:].split("]", 1)[0]
    return netloc.rsplit(":", 1)[0] if netloc.count(":") == 1 else netloc


def is_loopback(netloc: str) -> bool:
    return hostname(netloc) in LOOPBACK_HOSTS


def cookie_name(host: str) -> str:
    """rtool_session_8765 for Host 127.0.0.1:8765 (a host's cookies reach all its ports, so name them apart)."""
    netloc = host.strip().lower()
    port = netloc.rsplit(":", 1)[1] if netloc.count(":") == 1 or "]:" in netloc else ""
    return f"{SESSION_COOKIE}_{port}" if port.isdigit() else SESSION_COOKIE


def body_limit(path: str) -> int:
    return MAX_UPLOAD_BODY if path.endswith("/literature/inbox") else MAX_BODY


class SecurityMiddleware:
    def __init__(self, app: ASGIApp, *, token: str) -> None:
        self.app = app
        self.token = token
        self.session = session_value(token)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope.get("headers", [])}
        problem = self.check(scope["method"], scope["path"], headers)
        if problem is not None:
            status, detail = problem
            await self.reply(send, status, detail)
            return
        limit = body_limit(scope["path"])
        declared = headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            await self.reply(send, 413, f"the request is too large (at most {limit // (1024 * 1024)} MB)")
            return
        received = 0

        async def receive_capped() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:  # a body without (or with a false) Content-Length: stop reading it
                    return {"type": "http.disconnect"}
            return message

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                present = {k.lower() for k, _ in message.get("headers", [])}
                message["headers"] = list(message.get("headers", [])) + [
                    (k, v) for k, v in SECURITY_HEADERS if k not in present
                ]
            await send(message)

        await self.app(scope, receive_capped, send_with_headers)

    def authenticated(self, headers: dict[str, str]) -> bool:
        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer ") and hmac.compare_digest(auth[7:].strip(), self.token):
            return True
        cookie = SimpleCookie()
        try:
            cookie.load(headers.get("cookie", ""))
        except Exception:  # noqa: BLE001 - a malformed cookie header is just "not logged in"
            return False
        morsel = cookie.get(cookie_name(headers.get("host", "")))
        return morsel is not None and hmac.compare_digest(morsel.value, self.session)

    def check(self, method: str, path: str, headers: dict[str, str]) -> tuple[int, str] | None:
        if not is_loopback(headers.get("host", "")):
            return 400, "unexpected Host header; open the UI through 127.0.0.1 or localhost"
        if not path.startswith("/api/"):
            return None  # the app shell and its assets are not secret
        if method in MUTATING:
            if headers.get("x-requested-with") != "rtool":
                return 403, "missing X-Requested-With: rtool"
            origin = headers.get("origin")
            if origin and origin != "null" and not is_loopback(urlsplit(origin).netloc):
                return 403, "cross-site request refused"
        if path not in PUBLIC_PATHS and not self.authenticated(headers):
            return 401, "not signed in: open the link printed by `rtool ui` (it contains the access token)"
        return None

    @staticmethod
    async def reply(send: Send, status: int, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
                + SECURITY_HEADERS,
            }
        )
        await send({"type": "http.response.body", "body": body})
