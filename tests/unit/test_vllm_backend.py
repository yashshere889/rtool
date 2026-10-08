from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from research_tool.core.config import LLMConfig
from research_tool.llm.client import LLMClient
from research_tool.llm.types import ChatMessage, ChatRequest
from research_tool.llm.vllm_backend import LLMRequestError, VLLMBackend, apply_reasoning_flag, split_reasoning
from research_tool.testing.toy_agents import ToyNumbers

Handler = Callable[[httpx.Request], httpx.Response]


def make_backend(handler: Handler, **kwargs: Any) -> VLLMBackend:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return VLLMBackend(
        "http://srv:8000/v1", model="nemotron", api_key="secret", client=client, sleep=lambda _s: None, **kwargs
    )


def completion(content: str, *, reasoning: str | None = None, field: str = "reasoning_content") -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message[field] = reasoning
    return {
        "model": "nemotron",
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


def request(text: str = "hi", **kwargs: Any) -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role="user", content=text)], **kwargs)


def test_reasoning_flag_goes_into_the_system_prompt() -> None:
    assert apply_reasoning_flag([{"role": "user", "content": "q"}], "off")[0] == {
        "role": "system",
        "content": "/no_think",
    }
    with_system = apply_reasoning_flag(
        [{"role": "system", "content": "Be brief. /no_think"}, {"role": "user", "content": "q"}], "on"
    )
    assert with_system[0]["content"] == "Be brief.\n\n/think"  # existing flag replaced, not duplicated


def test_payload_and_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=completion("ok"))

    backend = make_backend(handler)
    schema = ToyNumbers.model_json_schema()
    backend.chat(request(top_p=0.95, stop=["END"], json_schema=schema, reasoning="on", temperature=0.6))
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/chat/completions"
    assert seen[0].headers["Authorization"] == "Bearer secret"
    assert body["model"] == "nemotron" and body["top_p"] == 0.95 and body["stop"] == ["END"]
    assert body["messages"][0] == {"role": "system", "content": "/think"}
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == schema

    for mode, key in (("structured_outputs", "structured_outputs"), ("guided_json", "guided_json")):
        seen.clear()
        make_backend(handler, structured_mode=mode).chat(request(json_schema=schema))
        assert key in json.loads(seen[0].content)
    seen.clear()
    make_backend(handler, reasoning_control="none").chat(request())
    assert json.loads(seen[0].content)["messages"][0]["role"] == "user"


@pytest.mark.parametrize(
    ("message", "expected_reasoning", "expected_text"),
    [
        (completion("answer", reasoning="because"), "because", "answer"),
        (completion("answer", reasoning="because", field="reasoning"), "because", "answer"),
        (completion("thinking hard</think>\n\nanswer"), "thinking hard", "answer"),  # template pre-filled <think>
        (completion("<think>still going"), "still going", ""),  # cut off while thinking
        (completion("plain answer"), None, "plain answer"),
    ],
)
def test_reasoning_extraction(message: dict[str, Any], expected_reasoning: str | None, expected_text: str) -> None:
    response = make_backend(lambda _r: httpx.Response(200, json=message)).chat(request())
    assert (response.reasoning, response.text) == (expected_reasoning, expected_text)
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (11, 7)


def test_split_reasoning_helper() -> None:
    assert split_reasoning("<think>a</think>b") == ("a", "b")
    assert split_reasoning("b") == (None, "b")


def test_transient_errors_are_retried() -> None:
    replies = iter(
        [
            httpx.Response(503, json={"error": "loading"}),
            httpx.ConnectError("refused"),
            httpx.Response(200, json=completion("ok")),
        ]
    )
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    assert make_backend(handler).chat(request()).text == "ok"
    assert len(calls) == 3


def test_client_errors_fail_fast_and_outages_give_up() -> None:
    calls = []

    def bad_request(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(400, json={"error": {"message": "maximum context length exceeded"}})

    with pytest.raises(LLMRequestError, match="maximum context length"):
        make_backend(bad_request).chat(request())
    assert len(calls) == 1
    with pytest.raises(LLMRequestError, match="after 3 attempts"):
        make_backend(lambda _r: httpx.Response(502, text="bad gateway"), max_retries=2).chat(request())


def test_chat_many_keeps_order_and_reports_errors() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        text = json.loads(req.content)["messages"][-1]["content"]
        if text == "bad":
            return httpx.Response(400, json={"error": "nope"})
        return httpx.Response(200, json=completion(text.upper()))

    results = make_backend(handler).chat_many([request("a"), request("bad"), request("c")], concurrency=3)
    assert [getattr(r, "text", None) for r in results] == ["A", None, "C"]
    assert isinstance(results[1], LLMRequestError)


def test_server_info_endpoints() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/health":
            return httpx.Response(200)
        if req.url.path == "/version":
            return httpx.Response(200, json={"version": "0.10.1"})
        if req.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 42, "tokens": []})
        return httpx.Response(404)

    backend = make_backend(handler)
    assert backend.health() and backend.server_version() == "0.10.1" and backend.count_tokens("x") == 42
    assert LLMClient(backend, LLMConfig()).count_tokens("whatever") == 42


def test_structured_through_the_client() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion('{"values": [1.0, 2.0]}', reasoning="checked the numbers"))

    client = LLMClient(make_backend(handler), LLMConfig())
    assert client.structured(ToyNumbers, "numbers please", profile="reason").values == [1.0, 2.0]


def test_two_phase_structured_when_server_cannot_mix_reasoning_and_json() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(200, json=completion('{"values": [3.0]}'))
        return httpx.Response(200, json=completion("I think the answer is three.", reasoning="..."))

    config = LLMConfig(structured_with_reasoning=False)
    result = LLMClient(make_backend(handler), config).structured(ToyNumbers, "numbers", profile="reason")
    assert result.values == [3.0]
    first, second = bodies
    assert "response_format" not in first and first["messages"][0]["content"] == "/think"
    assert "response_format" in second and second["messages"][0]["content"] == "/no_think"
    assert second["temperature"] == 0.0 and "top_p" not in second
