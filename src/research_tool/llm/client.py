"""The LLM facade agents use: profiles in, text or validated models out.

Agents never pick model ids or sampling parameters; they name a *profile*
(``extract``, ``reason``, ``code``, ``write``, ``judge``) from the config.
Every call is charged to the stage/run budget and reported as an event.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..core.budgets import BudgetTracker
from ..core.config import LLMConfig
from ..core.errors import BudgetExceeded, RToolError
from .schema_guide import require_all, schema_guide
from .types import ChatMessage, ChatRequest, ChatResponse, LLMBackend

M = TypeVar("M", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


class LLMOutputError(RToolError):
    """The model kept producing output that does not match the requested schema."""


def strip_reasoning(text: str) -> str:
    return _THINK.sub("", text).strip()


def extract_json(text: str) -> str:
    """Pull the JSON value out of a reply that may include fences or chatter."""
    text = strip_reasoning(text)
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    if text[:1] in "{[":
        return text
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    end = max(text.rfind("}"), text.rfind("]"))
    if not starts or end <= min(starts):
        raise ValueError("no JSON object found in model output")
    return text[min(starts) : end + 1]


def _as_messages(prompt: str | Sequence[ChatMessage], system: str | None) -> list[ChatMessage]:
    messages = [ChatMessage(role="user", content=prompt)] if isinstance(prompt, str) else list(prompt)
    if system is not None:
        messages = [ChatMessage(role="system", content=system), *messages]
    return messages


LOOP_BREAKER: dict[str, Any] = {"temperature": 0.6, "top_p": 0.95}
CONCISE = "Be concise: state each item once and never repeat yourself."


def _with_note(messages: list[ChatMessage], note: str) -> list[ChatMessage]:
    """``messages`` with ``note`` appended to the last user message."""
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].role == "user":
            out[i] = ChatMessage(role="user", content=f"{out[i].content}\n\n{note}")
            break
    return out


def _short(exc: Exception, limit: int = 600) -> str:
    text = str(exc).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


class LLMClient:
    def __init__(
        self,
        backend: LLMBackend,
        config: LLMConfig,
        *,
        budget: BudgetTracker | None = None,
        on_call: Callable[[dict[str, Any]], None] | None = None,
        stage: str | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.budget = budget
        self.on_call = on_call
        self.stage = stage

    def bind(
        self,
        *,
        budget: BudgetTracker | None,
        on_call: Callable[[dict[str, Any]], None] | None,
        stage: str | None,
    ) -> LLMClient:
        """A per-stage view that charges usage to that stage's budget."""
        return LLMClient(self.backend, self.config, budget=budget, on_call=on_call, stage=stage)

    def chat(
        self,
        prompt: str | Sequence[ChatMessage],
        *,
        profile: str = "extract",
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> ChatResponse:
        settings = self.config.profile(profile)
        request = ChatRequest(
            messages=_as_messages(prompt, system),
            profile=profile,
            reasoning=settings.reasoning,
            temperature=settings.temperature,
            top_p=settings.top_p,
            presence_penalty=settings.presence_penalty,
            frequency_penalty=settings.frequency_penalty,
            max_tokens=settings.max_tokens,
            thinking_budget=settings.thinking_budget,
            json_schema=json_schema,
            metadata={"stage": self.stage},
        )
        if overrides:
            request = request.model_copy(update=overrides)
        if self.budget is not None:
            problem = self.budget.would_exceed(llm_calls=1)
            if problem:
                raise BudgetExceeded(problem)
        started = time.monotonic()
        response = self.backend.chat(request)
        response.latency_s = response.latency_s or (time.monotonic() - started)
        if self.budget is not None:
            self.budget.charge(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                llm_calls=1,
            )
        if self.on_call is not None:
            self.on_call(
                {
                    "profile": profile,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "latency_s": round(response.latency_s, 3),
                    "finish_reason": response.finish_reason,
                    "structured": json_schema is not None,
                }
            )
        return response

    def structured(
        self,
        model: type[M],
        prompt: str | Sequence[ChatMessage],
        *,
        profile: str = "extract",
        system: str | None = None,
        retries: int = 2,
    ) -> M:
        """Ask for JSON matching ``model``; on invalid output, show the error and retry."""
        schema = model.model_json_schema()
        messages = _as_messages(prompt, system)
        if self.config.describe_schema:
            # The server's constraint never shows the model the field descriptions, and lets it skip
            # any field with a default: spell the fields out, and make the constraint ask for all of them.
            guide = schema_guide(schema)
            messages = _with_note(messages, guide) if guide else messages
            schema = require_all(schema)
        overrides: dict[str, Any] = {}
        if self.config.profile(profile).reasoning == "on" and not self.config.structured_with_reasoning:
            # The server can't constrain output after free-form reasoning: think first, then format.
            draft = self.chat(messages, profile=profile)
            messages = [
                *messages,
                ChatMessage(role="assistant", content=draft.text or "(no answer)"),
                ChatMessage(
                    role="user", content="Now give only the final answer as JSON matching the required schema."
                ),
            ]
            overrides = {"reasoning": "off", "temperature": 0.0, "top_p": None}
        last_error: Exception | None = None
        base = messages
        for _ in range(retries + 1):
            response = self.chat(messages, profile=profile, json_schema=schema, **overrides)
            try:
                return model.model_validate_json(extract_json(response.text))
            except (ValidationError, ValueError) as exc:
                last_error = exc
                if response.finish_reason == "length":
                    # Out of tokens mid-JSON: nearly always greedy decoding repeating itself (seen on
                    # Barkla: the same item or sentence until max_tokens). Showing that back wastes
                    # the context; ask again, briefly, with the model card's sampling settings.
                    messages = _with_note(base, CONCISE)
                    overrides = {**overrides, **LOOP_BREAKER}
                    continue
                messages = [
                    *messages,
                    ChatMessage(role="assistant", content=response.text),
                    ChatMessage(
                        role="user",
                        content=(
                            f"That reply did not match the required JSON schema ({_short(exc)}). "
                            "Reply again with only the corrected JSON."
                        ),
                    ),
                ]
        raise LLMOutputError(
            f"output did not match {model.__name__} after {retries + 1} attempts: {_short(last_error or Exception())}"
        )

    # -------------------------------------------------------------- bulk work
    def parallel(self, fn: Callable[[Any], Any], items: Sequence[Any], concurrency: int | None = None) -> list[Any]:
        """Apply ``fn`` (which makes LLM calls) to items concurrently so the server can batch them.

        Results keep the input order; an item that raised is returned as its exception,
        except ``BudgetExceeded``, which stops the whole batch.
        """

        def one(item: Any) -> Any:
            try:
                return fn(item)
            except BudgetExceeded:
                raise  # stops the batch: the stage is out of budget
            except Exception as exc:  # noqa: BLE001 - reported per item, the caller decides
                return exc

        workers = max(1, concurrency or self.config.concurrency)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, items))

    def structured_many(
        self,
        model: type[M],
        prompts: Sequence[str | Sequence[ChatMessage]],
        *,
        profile: str = "extract",
        system: str | None = None,
        retries: int = 2,
        concurrency: int | None = None,
    ) -> list[M | Exception]:
        """``structured`` over many prompts concurrently (the server batches them).

        Results keep the input order; a failed item is returned as its exception.
        """
        return self.parallel(
            lambda p: self.structured(model, p, profile=profile, system=system, retries=retries), prompts, concurrency
        )

    def chat_many(
        self,
        prompts: Sequence[str | Sequence[ChatMessage]],
        *,
        profile: str = "extract",
        system: str | None = None,
        concurrency: int | None = None,
    ) -> list[ChatResponse | Exception]:
        return self.parallel(lambda p: self.chat(p, profile=profile, system=system), prompts, concurrency)

    def count_tokens(self, text: str) -> int:
        """Exact count from the server when it can tell us, else a ~4 chars/token estimate."""
        counter = getattr(self.backend, "count_tokens", None)
        if counter is not None:
            exact = counter(text)
            if exact is not None:
                return int(exact)
        return max(1, len(text) // 4)
