"""Helpers shared by the codeagent tests (vendored from coding-agent 5c04c9e)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from research_tool.codeagent import AssistantTurn, ToolCall


def run(coro):
    return asyncio.run(coro)


def call(name: str, i: int = 0, **args: Any) -> ToolCall:
    return ToolCall(id=f"call_{name}_{i}", name=name, arguments=json.dumps(args))


class ScriptedModel:
    """A fake ChatModel that plays back a list of turns and records what it was sent."""

    def __init__(self, turns: list[AssistantTurn]) -> None:
        self.turns = list(turns)
        self.requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    async def complete(self, messages, tools):
        self.requests.append(([dict(m) for m in messages], tools))
        return self.turns.pop(0)


def turn(text: str = "", calls: list[ToolCall] | None = None) -> AssistantTurn:
    calls = calls or []
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in calls
        ]
    return AssistantTurn(text, calls, "tool_calls" if calls else "stop", message)
