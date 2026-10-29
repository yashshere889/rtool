from __future__ import annotations

import pytest

from research_tool.core.budgets import BudgetTracker
from research_tool.core.config import BudgetLimits, LLMConfig
from research_tool.core.errors import BudgetExceeded
from research_tool.llm.client import LLMClient, LLMOutputError, extract_json
from research_tool.llm.mock_backend import MockLLM, example_from_schema
from research_tool.llm.types import ChatResponse
from research_tool.llm.vllm_backend import VLLMBackend
from research_tool.testing.toy_agents import ToyNumbers, ToyStats


def test_extract_json_handles_fences_chatter_and_reasoning() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('Sure! Here it is: {"a": [1, 2]} hope that helps') == '{"a": [1, 2]}'
    assert extract_json('<think>{"draft": 0}</think>{"a": 1}') == '{"a": 1}'
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_structured_repairs_invalid_output() -> None:
    replies = iter(['{"values": "oops"}', '{"values": [1.5, 2.5]}'])
    llm = MockLLM().add(lambda req: True, lambda req: next(replies))
    client = LLMClient(llm, LLMConfig())
    result = client.structured(ToyNumbers, "give numbers")
    assert result.values == [1.5, 2.5]
    assert len(llm.calls) == 2
    assert "did not match" in llm.calls[1].messages[-1].content  # the repair prompt shows the error
    assert llm.calls[0].json_schema is not None


def test_structured_breaks_repetition_loops_by_sampling() -> None:
    loop = ChatResponse(text='{"values": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0', finish_reason="length")
    replies = iter([loop, '{"values": [1.0, 2.0]}'])
    llm = MockLLM().add(lambda req: True, lambda req: next(replies))
    result = LLMClient(llm, LLMConfig()).structured(ToyNumbers, "give numbers")
    assert result.values == [1.0, 2.0]
    first, second = llm.calls
    assert first.temperature == 0.0 and (second.temperature, second.top_p) == (0.6, 0.95)
    assert len(second.messages) == len(first.messages)  # the looping reply is not shown back
    assert second.messages[-1].content.endswith("never repeat yourself.")


def test_penalties_reach_the_server_payload() -> None:
    profile = LLMConfig().profile("extract").model_copy(update={"presence_penalty": 0.3})
    config = LLMConfig(profiles={"extract": profile})
    llm = MockLLM()
    LLMClient(llm, config).chat("hi")
    assert llm.calls[0].presence_penalty == 0.3 and llm.calls[0].frequency_penalty is None
    payload = VLLMBackend("http://x/v1", model="m").build_payload(llm.calls[0])
    assert payload["presence_penalty"] == 0.3 and "frequency_penalty" not in payload


def test_structured_gives_up_after_retries() -> None:
    llm = MockLLM().add(lambda req: True, "not json at all")
    with pytest.raises(LLMOutputError, match="ToyNumbers"):
        LLMClient(llm, LLMConfig()).structured(ToyNumbers, "x", retries=1)
    assert len(llm.calls) == 2


def test_profiles_and_budget_charging() -> None:
    llm = MockLLM()
    budget = BudgetTracker(BudgetLimits(max_llm_calls=1), BudgetLimits())
    calls: list[dict] = []
    client = LLMClient(llm, LLMConfig()).bind(budget=budget, on_call=calls.append, stage="s")
    client.chat("hello", profile="reason")
    request = llm.calls[0]
    assert (request.reasoning, request.temperature, request.top_p) == ("on", 0.6, 0.95)
    assert budget.stage_usage.llm_calls == 1 and budget.stage_usage.prompt_tokens > 0
    assert calls and calls[0]["profile"] == "reason"
    with pytest.raises(BudgetExceeded):
        client.chat("again")  # refused before reaching the backend
    assert len(llm.calls) == 1


def test_example_from_schema_is_valid_for_models() -> None:
    for model in (ToyNumbers, ToyStats):
        model.model_validate(example_from_schema(model.model_json_schema()))


def test_structured_prompts_spell_out_the_fields_and_require_them_all() -> None:
    """The server's constraint never shows the model a field's description, and lets it skip defaulted fields:
    on 2026-10-12 that left 88 of 98 paper extractions without a method."""
    from research_tool.schemas.literature import PaperExtraction

    llm = MockLLM()
    LLMClient(llm, LLMConfig()).structured(PaperExtraction, "Extract the facts.\n<<<\npaper\n>>>")
    request = llm.calls[0]
    prompt = request.messages[-1].content
    assert prompt.startswith("Extract the facts.\n<<<\npaper\n>>>")  # the guide goes after the task
    assert '- "methods" (list of texts): Names of the methods' in prompt
    assert '- "key_results" (list of up to 8 objects, each with:)' in prompt
    assert '  - "quote" (text): Sentence copied verbatim from the paper that supports the claim' in prompt
    schema = request.json_schema or {}
    assert schema["required"] == list(schema["properties"])
    assert schema["$defs"]["KeyResult"]["required"] == ["claim", "quote", "method", "dataset", "metric", "value"]

    plain = MockLLM()
    LLMClient(plain, LLMConfig(describe_schema=False)).structured(PaperExtraction, "Extract the facts.")
    assert plain.calls[0].messages[-1].content == "Extract the facts."
    assert "methods" not in (plain.calls[0].json_schema or {}).get("required", [])


def test_schema_guide_describes_types_choices_and_bounds() -> None:
    from research_tool.llm.schema_guide import schema_guide
    from research_tool.schemas.literature import RelevanceJudgment, SearchPlan

    guide = schema_guide(RelevanceJudgment.model_json_schema())
    lines = guide.splitlines()[1:]
    assert [line.split('"')[1] for line in lines if line.startswith("- ")] == [
        "reason", "concepts", "subtopics", "out_of_scope", "score"
    ]  # fmt: skip
    assert '  - "match" ("yes" | "partly" | "no"): yes = the paper is about it' in guide
    assert '- "subtopics" (list of whole numbers)' in guide and '- "out_of_scope" (true or false)' in guide
    assert '- "score" (whole number 0-10)' in guide
    assert '- "year_from" (whole number, or null)' in schema_guide(SearchPlan.model_json_schema())
