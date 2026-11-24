"""The agent loop: ask the model, run the tools it asks for, send the results back, repeat."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

from .executor import ToolExecutor
from .llm import AssistantTurn, ChatModel
from .registry import ToolRegistry
from .tool import ToolContext

DEFAULT_SYSTEM_PROMPT = """You are a coding agent working in the project at {workspace}.

You act by calling tools. Independent read-only calls (read_file, glob, grep) can go in the same turn; they run in parallel.
- Read a file with read_file before changing it. edit_file replaces one exact, unique string; copy it from the file without the line-number prefix.
- Use glob and grep to find code; use bash for builds, tests, git and package managers.
- When a tool returns an error, read it and change your approach. Do not repeat a call that failed the same way.
- If a permission is denied, do not try to get around it. Do other useful work or explain what you need.
- After changing code, run the relevant tests or checks if there are any.
When the task is done, reply with a short summary of what you changed and do not call any tool."""


class Agent:
    def __init__(
        self,
        model: ChatModel,
        registry: ToolRegistry,
        executor: ToolExecutor,
        ctx: ToolContext,
        *,
        system_prompt: str | None = None,
        max_turns: int = 50,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.model = model
        self.registry = registry
        self.executor = executor
        self.ctx = ctx
        self.max_turns = max_turns
        self.on_event = on_event
        prompt = (system_prompt or DEFAULT_SYSTEM_PROMPT).replace("{workspace}", str(ctx.workspace))
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": prompt}]

    async def send(self, user_text: str) -> str:
        """Run one user request to completion and return the model's final text."""
        self.messages.append({"role": "user", "content": user_text})
        for _ in range(self.max_turns):
            if self.ctx.cancel.is_set():
                return "[interrupted]"
            turn = await self._complete_or_cancel()
            if turn is None:
                return "[interrupted]"
            self.messages.append(turn.message)
            if turn.text:
                self._emit({"type": "assistant_text", "text": turn.text})

            if not turn.tool_calls:
                if turn.finish_reason == "length":
                    return turn.text + "\n[stopped: the model hit max_tokens]"
                return turn.text

            outcomes = await self.executor.run(turn.tool_calls, self.ctx)
            # One tool message per call, in the order the model made them.
            for outcome in outcomes:
                content = outcome.result.content
                if outcome.result.is_error:
                    content = f"Error: {content}"
                self.messages.append(
                    {"role": "tool", "tool_call_id": outcome.call.id, "name": outcome.call.name, "content": content}
                )
        return f"[stopped: reached max_turns={self.max_turns}]"

    async def _complete_or_cancel(self) -> AssistantTurn | None:
        """Ask the model, but give up (and drop the request) as soon as the user cancels."""
        request = asyncio.create_task(self.model.complete(self.messages, self.registry.openai_tools(self.ctx)))
        cancelled = asyncio.create_task(self.ctx.cancel.wait())
        try:
            done, _ = await asyncio.wait({request, cancelled}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            cancelled.cancel()
        if request in done:
            return request.result()
        request.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await request
        return None

    def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(event)
