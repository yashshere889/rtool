from __future__ import annotations

import json
import re

import httpx

from research_tool.core.config import LLMConfig
from research_tool.llm.bench import recommendations, run_bench
from research_tool.llm.client import LLMClient
from research_tool.llm.mock_backend import example_from_schema
from research_tool.llm.vllm_backend import VLLMBackend


def fake_vllm(req: httpx.Request) -> httpx.Response:
    """A well-behaved server: valid JSON when asked, separate reasoning, perfect recall."""
    path = req.url.path
    if path == "/health":
        return httpx.Response(200)
    if path == "/version":
        return httpx.Response(200, json={"version": "0.10.1"})
    if path == "/tokenize":
        return httpx.Response(200, json={"count": len(json.loads(req.content)["prompt"]) // 4})
    body = json.loads(req.content)
    last = body["messages"][-1]["content"]
    reasoning = "thinking" if body["messages"][0]["content"].endswith("/think") else None
    if "response_format" in body:
        content = json.dumps(example_from_schema(body["response_format"]["json_schema"]["schema"]))
    elif "secret project code" in last:
        content = re.search(r"secret project code is (\w+)\.", last).group(1)  # type: ignore[union-attr]
    else:
        content = "One. Two. Three."
    message = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    usage = {"prompt_tokens": len(last) // 4, "completion_tokens": 12}
    return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop"}], "usage": usage})


def test_bench_reports_a_healthy_server() -> None:
    backend = VLLMBackend(
        "http://srv/v1", model="nemotron", client=httpx.Client(transport=httpx.MockTransport(fake_vllm))
    )
    report = run_bench(
        LLMClient(backend, LLMConfig()), structured_n=6, context_sizes=(500, 2000), concurrency_levels=(1, 4)
    )
    assert report["healthy"] and report["server_version"] == "0.10.1"
    assert report["structured_reasoning_off"]["first_try_rate"] == 1.0
    assert report["structured_reasoning_on"]["valid_after_repair"] == report["structured_reasoning_on"]["n"]
    assert report["reasoning"]["reasoning_returned_separately"]
    assert not report["reasoning"]["think_tags_leaked_into_answer"]
    assert all(case["found"] for case in report["long_context"])
    assert [t["concurrency"] for t in report["throughput"]] == [1, 4]
    assert any("Best throughput" in tip for tip in report["recommendations"])
    assert not any("unreliable" in tip for tip in report["recommendations"])


def test_recommendations_flag_problems() -> None:
    tips = recommendations(
        {
            "structured_reasoning_off": {"n": 10, "first_try_valid": 6, "valid_after_repair": 9, "first_try_rate": 0.6},
            "structured_reasoning_on": {"n": 10, "first_try_valid": 2, "valid_after_repair": 5, "first_try_rate": 0.2},
            "reasoning": {"think_tags_leaked_into_answer": True},
            "long_context": [{"target_tokens": 8000, "found": True}, {"target_tokens": 32000, "found": False}],
            "throughput": [],
        }
    )
    text = " ".join(tips)
    assert "structured_mode" in text and "structured_with_reasoning: false" in text
    assert "reasoning parser" in text and "under ~8000 tokens" in text
