"""A deterministic stand-in for the real model (tests, dry runs, CI)."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from .types import ChatRequest, ChatResponse, TokenUsage

Reply = str | dict[str, Any] | list[Any] | BaseModel | ChatResponse
Matcher = str | Callable[[ChatRequest], bool]
Handler = Reply | Callable[[ChatRequest], Reply]


def example_from_schema(schema: dict[str, Any], root: dict[str, Any] | None = None) -> Any:
    """Smallest value that satisfies a (Pydantic-generated) JSON Schema."""
    root = root if root is not None else schema
    if "$ref" in schema:
        target: Any = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        return example_from_schema(target, root)
    if "default" in schema:
        return copy.deepcopy(schema["default"])
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            options = [s for s in schema[key] if s.get("type") != "null"] or schema[key]
            return example_from_schema(options[0], root)
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "null")
    if kind == "object" or "properties" in schema:
        props = schema.get("properties", {})
        required = schema.get("required", [])
        return {name: example_from_schema(props[name], root) for name in required if name in props}
    if kind == "array":
        # One element by default so dry runs exercise the code that consumes lists.
        count = max(schema.get("minItems", 1), 1) if schema.get("maxItems", 1) > 0 else 0
        return [example_from_schema(schema.get("items", {}), root) for _ in range(count)]
    if kind == "string":
        return "x" * max(schema.get("minLength", 7), 1) if schema.get("minLength") else "example"
    if kind == "integer":
        return int(schema.get("minimum", 0))
    if kind == "number":
        return float(schema.get("minimum", 0))
    if kind == "boolean":
        return False
    return None


class MockLLM:
    """Rule-based fake model.

    Rules are tried in order; a rule matches when its string occurs in the last
    message (or its predicate returns True). Unmatched structured requests get a
    schema-shaped placeholder; unmatched text requests get an echo.
    """

    def __init__(self, rules: list[tuple[Matcher, Handler]] | None = None) -> None:
        self.rules: list[tuple[Matcher, Handler]] = list(rules or [])
        self.calls: list[ChatRequest] = []

    def add(self, match: Matcher, reply: Handler) -> MockLLM:
        self.rules.append((match, reply))
        return self

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        last = request.messages[-1].content if request.messages else ""
        for match, handler in self.rules:
            hit = match(request) if callable(match) else match in last
            if hit:
                reply = handler(request) if callable(handler) else handler
                return self._respond(reply, request)
        if request.json_schema is not None:
            return self._respond(example_from_schema(request.json_schema), request)
        return self._respond(f"[mock reply to: {last[:80]}]", request)

    @staticmethod
    def _respond(reply: Reply, request: ChatRequest) -> ChatResponse:
        if isinstance(reply, ChatResponse):
            return reply
        if isinstance(reply, BaseModel):
            text = reply.model_dump_json()
        elif isinstance(reply, dict | list):
            text = json.dumps(reply)
        else:
            text = str(reply)
        prompt_chars = sum(len(m.content) for m in request.messages)
        usage = TokenUsage(prompt_tokens=max(1, prompt_chars // 4), completion_tokens=max(1, len(text) // 4))
        return ChatResponse(text=text, usage=usage, model="mock")
