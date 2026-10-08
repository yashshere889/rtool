"""HTTP backend for vLLM's OpenAI-compatible server (works with other compatible servers too).

Model-specific behaviour handled here:

* **Reasoning control** — Nemotron Nano v2 switches reasoning with ``/think`` or
  ``/no_think`` in the system prompt (model card). We set exactly one flag per call.
* **Reasoning extraction** — with ``--reasoning-parser`` the server returns the
  thinking in ``reasoning_content`` (older vLLM) or ``reasoning`` (newer); without
  it, the text contains ``...</think>answer`` (the chat template pre-fills ``<think>``).
* **Structured output** — JSON Schema constraints are sent as OpenAI
  ``response_format`` by default; the legacy ``guided_json`` and the newer
  ``structured_outputs`` fields are selectable because vLLM changed them across
  releases.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from ..core.errors import RToolError
from .types import ChatRequest, ChatResponse, TokenUsage

REASONING_FLAGS = {"on": "/think", "off": "/no_think"}
_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


class LLMRequestError(RToolError):
    """The server rejected the request (4xx) or stayed unavailable after retries."""


def apply_reasoning_flag(messages: list[dict[str, str]], reasoning: str) -> list[dict[str, str]]:
    """Put exactly one of /think or /no_think at the end of the system message."""
    flag = REASONING_FLAGS[reasoning]
    out = [dict(m) for m in messages]
    if out and out[0]["role"] == "system":
        content = out[0]["content"].replace("/no_think", "").replace("/think", "").rstrip()
        out[0]["content"] = f"{content}\n\n{flag}" if content else flag
    else:
        out.insert(0, {"role": "system", "content": flag})
    return out


def split_reasoning(text: str) -> tuple[str | None, str]:
    """Separate ``<think>`` reasoning from the answer in raw model text."""
    if "</think>" in text:
        reasoning, answer = text.split("</think>", 1)
        return reasoning.replace("<think>", "").strip() or None, answer.strip()
    if "<think>" in text:  # never closed: generation stopped while still thinking
        before, reasoning = text.split("<think>", 1)
        return reasoning.strip() or None, before.strip()
    return None, text.strip()


def _error_text(response: httpx.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            error = data.get("error") or data.get("detail") or data.get("message") or data
            if isinstance(error, dict):
                error = error.get("message") or error
            return str(error)[:500]
    except ValueError:
        pass
    return response.text[:500]


class VLLMBackend:
    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 900,
        max_retries: int = 3,
        structured_mode: str = "auto",
        reasoning_control: str = "system_flag",
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        root = base_url.rstrip("/")
        self.root = root[: -len("/v1")] if root.endswith("/v1") else root
        self.api = self.root + "/v1"
        self.model = model
        self.max_retries = max_retries
        self.structured_mode = "response_format" if structured_mode == "auto" else structured_mode
        self.reasoning_control = reasoning_control
        self._sleep = sleep
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = client or httpx.Client(timeout=httpx.Timeout(timeout_s, connect=15.0), headers=headers)
        if client is not None and api_key:
            self._http.headers.update(headers)

    # ---------------------------------------------------------------- requests
    def _structured_fields(self, schema: dict[str, Any]) -> dict[str, Any]:
        if self.structured_mode == "response_format":
            name = _NAME_UNSAFE.sub("_", str(schema.get("title") or "output"))[:64]
            return {"response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}}
        if self.structured_mode == "structured_outputs":
            return {"structured_outputs": {"json": schema}}
        if self.structured_mode == "guided_json":
            return {"guided_json": schema}
        return {}

    def build_payload(self, request: ChatRequest) -> dict[str, Any]:
        messages = [m.model_dump() for m in request.messages]
        if self.reasoning_control == "system_flag":
            messages = apply_reasoning_flag(messages, request.reasoning)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        for name in ("presence_penalty", "frequency_penalty"):
            if getattr(request, name) is not None:
                payload[name] = getattr(request, name)
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.json_schema is not None:
            payload.update(self._structured_fields(request.json_schema))
        return payload

    def _request(self, method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        delay = 1.0
        problem = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = self._http.request(method, url, json=payload)
            except httpx.TransportError as exc:
                problem = f"connection error: {exc}"
            else:
                if response.status_code < 400:
                    return response.json()
                if response.status_code < 500 and response.status_code not in (408, 429):
                    raise LLMRequestError(
                        f"server rejected the request (HTTP {response.status_code}): {_error_text(response)}"
                    )
                problem = f"HTTP {response.status_code}: {_error_text(response)}"
            if attempt < self.max_retries:
                self._sleep(delay)
                delay = min(delay * 2, 30.0)
        raise LLMRequestError(f"LLM server unavailable after {self.max_retries + 1} attempts ({problem})")

    def chat(self, request: ChatRequest) -> ChatResponse:
        started = time.monotonic()
        data = self._request("POST", self.api + "/chat/completions", self.build_payload(request))
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError):
            raise LLMRequestError(f"unexpected response shape: {str(data)[:300]}") from None
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning is None:
            reasoning, content = split_reasoning(content)
        else:
            content = content.replace("</think>", "").strip()
        usage = data.get("usage") or {}
        return ChatResponse(
            text=content,
            reasoning=reasoning,
            finish_reason=choice.get("finish_reason") or "stop",
            usage=TokenUsage(
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
            ),
            model=str(data.get("model") or self.model),
            latency_s=time.monotonic() - started,
        )

    def chat_many(self, requests: Sequence[ChatRequest], concurrency: int = 16) -> list[ChatResponse | Exception]:
        """Send requests concurrently (vLLM batches them); results keep the input order."""

        def one(request: ChatRequest) -> ChatResponse | Exception:
            try:
                return self.chat(request)
            except Exception as exc:  # noqa: BLE001 - reported per item
                return exc

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            return list(pool.map(one, requests))

    # ------------------------------------------------------------ server info
    def health(self) -> bool:
        try:
            return self._http.get(self.root + "/health", timeout=10).status_code == 200
        except httpx.TransportError:
            return False

    def server_version(self) -> str | None:
        try:
            response = self._http.get(self.root + "/version", timeout=10)
            return str(response.json().get("version")) if response.status_code == 200 else None
        except (httpx.TransportError, ValueError, AttributeError):
            return None

    def count_tokens(self, text: str) -> int | None:
        """Exact token count via the server's /tokenize endpoint (None if unavailable)."""
        try:
            response = self._http.post(self.root + "/tokenize", json={"model": self.model, "prompt": text}, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return int(data.get("count") or len(data.get("tokens") or []))
        except (httpx.TransportError, ValueError):
            return None
        return None

    def close(self) -> None:
        self._http.close()
