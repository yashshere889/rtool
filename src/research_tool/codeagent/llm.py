"""Model adapters. The agent only needs ``ChatModel.complete(messages, tools) -> AssistantTurn``.

``RToolModel`` sends every turn through rtool's ``LLMClient`` (``ctx.llm``), so profiles, the stage's
budget, ``llm_call`` events and the backend's retries apply exactly as for any other rtool call.
(The original package's ``OpenAICompatibleModel`` is not vendored: rtool has no ``openai`` dependency.)

The vLLM server must run with tool calling turned on, for example
``--enable-auto-tool-choice --tool-call-parser qwen3_coder`` (``llm.server.extra_args``); without a
working parser the model's tool calls arrive as plain text and nothing runs.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from ..llm.types import ChatMessage
from .executor import ToolCall

if TYPE_CHECKING:
    from ..llm.client import LLMClient

_MESSAGE_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall]
    finish_reason: str | None
    message: dict[str, Any]  # the assistant message to append to the history, as-is
    usage: dict[str, Any] = field(default_factory=dict)


class ChatModel(Protocol):
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AssistantTurn: ...


class RToolModel:
    """A ``ChatModel`` backed by rtool's (synchronous) ``LLMClient``."""

    def __init__(self, llm: LLMClient, *, profile: str = "code", **overrides: Any) -> None:
        self.llm, self.profile, self.overrides = llm, profile, overrides

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AssistantTurn:
        history = [ChatMessage(**{k: m[k] for k in _MESSAGE_KEYS if m.get(k) is not None}) for m in messages]
        reply = await asyncio.to_thread(
            self.llm.chat, history, profile=self.profile, tools=tools or None, **self.overrides
        )
        calls = []
        for tc in reply.tool_calls:
            fn = tc.get("function") or {}
            if not fn.get("name"):
                continue
            args = fn.get("arguments")
            args = args if isinstance(args, str) else json.dumps(args or {})
            calls.append(ToolCall(id=tc.get("id") or f"call_{uuid.uuid4().hex[:16]}", name=fn["name"], arguments=args))
        message: dict[str, Any] = {"role": "assistant", "content": reply.text}
        if calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in calls
            ]
        return AssistantTurn(reply.text, calls, reply.finish_reason, message, reply.usage.model_dump())
