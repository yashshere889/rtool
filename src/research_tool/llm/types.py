"""Request/response types shared by every LLM backend."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    profile: str = "extract"
    reasoning: Literal["off", "on"] = "off"
    temperature: float = 0.0
    top_p: float | None = None
    presence_penalty: float | None = None  # both count generated tokens only (never the prompt,
    frequency_penalty: float | None = None  # so quotes can still be copied from it)
    max_tokens: int = 2048
    thinking_budget: int | None = None
    json_schema: dict[str, Any] | None = None  # constrain the answer to this JSON Schema
    stop: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)  # stage, purpose... (logging only)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatResponse(BaseModel):
    text: str  # the answer, with any reasoning removed
    reasoning: str | None = None
    finish_reason: str = "stop"
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = ""
    latency_s: float = 0.0


class LLMBackend(Protocol):
    """Anything that can answer a ChatRequest (mock, vLLM over HTTP, ...)."""

    def chat(self, request: ChatRequest) -> ChatResponse: ...
