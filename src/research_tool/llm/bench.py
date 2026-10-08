"""`rtool llm-bench`: measure what the agents rely on, on the real served model.

Checks (each one maps to a config decision):

* structured output validity, reasoning off  -> is ``structured_mode`` right?
* structured output validity, reasoning on   -> ``structured_with_reasoning`` true or false?
* reasoning returned separately from answer  -> is the server's reasoning parser working?
* long-context recall (needle in filler)     -> safe prompt size for map-reduce steps
* batched throughput at several concurrencies -> ``llm.concurrency``
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.fs import now_iso
from .client import LLMClient, LLMOutputError, extract_json
from .types import ChatMessage


class PaperFacts(BaseModel):
    title: str
    year: int
    open_access: bool


class Finding(BaseModel):
    claim: str
    evidence_quote: str
    confidence: Literal["low", "medium", "high"]


class Findings(BaseModel):
    findings: list[Finding] = Field(min_length=1)


class HypothesisSketch(BaseModel):
    statement: str
    independent_variables: list[str]
    dependent_variables: list[str]
    metric: str
    falsifiable: bool


_ABSTRACTS = [
    (
        "Sparse Mixture Routing for Low-Resource Translation",
        2023,
        True,
        "We show that sparse expert routing improves BLEU by 2.1 points on four low-resource language pairs "
        "while using 40% less compute than dense baselines. Gains vanish when training data exceeds 10M sentences.",
    ),
    (
        "Sleep Spindle Density Predicts Overnight Memory Consolidation",
        2021,
        False,
        "In 64 adults, spindle density during N2 sleep correlated with overnight recall improvement (r = 0.41). "
        "Pharmacologically suppressing spindles abolished the effect, suggesting a causal role.",
    ),
    (
        "Graph Neural Surrogates for Ocean Heat Transport",
        2024,
        True,
        "A message-passing surrogate reproduces meridional heat transport within 6% of the full model at 300x "
        "lower cost, but drifts after 40 simulated years without periodic re-anchoring.",
    ),
]

_CASES: list[tuple[type[BaseModel], str]] = [
    (
        PaperFacts,
        "Extract the title, publication year and whether it is open access.\n\n"
        "Title: {title}\nYear: {year}\nOpen access: {oa}\nAbstract: {abstract}",
    ),
    (
        Findings,
        "List the key findings of this abstract. Each evidence_quote must be copied verbatim "
        "from the abstract.\n\nAbstract: {abstract}",
    ),
    (HypothesisSketch, "Propose one testable follow-up hypothesis for this work.\n\nAbstract: {abstract}"),
]


def _structured_prompts(n: int) -> list[tuple[type[BaseModel], str]]:
    prompts = []
    for i in range(n):
        model, template = _CASES[i % len(_CASES)]
        title, year, oa, abstract = _ABSTRACTS[(i // len(_CASES)) % len(_ABSTRACTS)]
        prompts.append((model, template.format(title=title, year=year, oa=oa, abstract=abstract)))
    return prompts


def _first_try_validity(client: LLMClient, items: list[tuple[type[BaseModel], str]], profile: str) -> dict[str, Any]:
    """How often is the *first* answer valid (no repair)? Plus validity after the client's repair loop."""

    failures: list[dict[str, Any]] = []

    def first_try(item: tuple[type[BaseModel], str]) -> bool:
        model, prompt = item
        response = client.chat(prompt, profile=profile, json_schema=model.model_json_schema())
        try:
            model.model_validate_json(extract_json(response.text))
            return True
        except ValueError as exc:
            failures.append(
                {
                    "schema": model.__name__,
                    "finish_reason": response.finish_reason,
                    "completion_tokens": response.usage.completion_tokens,
                    "error": str(exc).splitlines()[0][:160],
                    "head": response.text[:160],
                    "tail": response.text[-160:],
                }
            )
            return False

    def repaired(item: tuple[type[BaseModel], str]) -> bool:
        model, prompt = item
        try:
            client.structured(model, prompt, profile=profile, retries=2)
            return True
        except LLMOutputError:
            return False

    first = client.parallel(first_try, items, None)
    after = client.parallel(repaired, items, None)
    n = len(items)
    first_ok = sum(1 for r in first if r is True)
    after_ok = sum(1 for r in after if r is True)
    errors = [str(r)[:200] for r in (*first, *after) if isinstance(r, Exception)][:5]
    return {
        "n": n,
        "first_try_valid": first_ok,
        "valid_after_repair": after_ok,
        "first_try_rate": round(first_ok / n, 3) if n else None,
        "errors": errors,
        "invalid_samples": failures[:5],
    }


def _needle(client: LLMClient, target_tokens: int, rng: random.Random) -> dict[str, Any]:
    code = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))
    filler = ("The committee reviewed routine maintenance logs and found nothing unusual. " * 40).strip()
    per_chunk = max(1, client.count_tokens(filler))
    chunks = [filler] * max(1, target_tokens // per_chunk)
    depth = rng.randint(0, len(chunks) - 1)
    chunks.insert(depth, f"Note for the archive: the secret project code is {code}.")
    document = " ".join(chunks)
    started = time.monotonic()
    try:
        answer = client.chat(
            [
                ChatMessage(
                    role="user", content=f"{document}\n\nWhat is the secret project code? Reply with the code only."
                )
            ],
            profile="extract",
            max_tokens=32,
        ).text
    except Exception as exc:  # noqa: BLE001 - a context-length error is a result here
        return {"target_tokens": target_tokens, "found": False, "error": str(exc)[:200]}
    return {
        "target_tokens": target_tokens,
        "actual_tokens": client.count_tokens(document),
        "found": code in answer,
        "latency_s": round(time.monotonic() - started, 2),
    }


def _throughput(client: LLMClient, concurrency: int) -> dict[str, Any]:
    prompts = [f"Write three short sentences about topic number {i}." for i in range(concurrency * 2)]
    started = time.monotonic()
    results = client.parallel(lambda p: client.chat(p, profile="extract", max_tokens=128), prompts, concurrency)
    wall = time.monotonic() - started
    tokens = sum(r.usage.completion_tokens for r in results if not isinstance(r, Exception))
    return {
        "concurrency": concurrency,
        "requests": len(prompts),
        "wall_s": round(wall, 2),
        "completion_tokens": tokens,
        "tokens_per_s": round(tokens / wall, 1) if wall > 0 else None,
        "errors": sum(1 for r in results if isinstance(r, Exception)),
    }


def run_bench(
    client: LLMClient,
    *,
    structured_n: int = 30,
    reasoning_n: int | None = None,  # structured prompts with reasoning on (default: a third); 0 skips
    context_sizes: tuple[int, ...] = (8_000, 32_000, 60_000),
    concurrency_levels: tuple[int, ...] = (1, 8, 32),
    seed: int = 7,
    progress: Callable[[str], None] = lambda _msg: None,
) -> dict[str, Any]:
    backend = client.backend
    report: dict[str, Any] = {
        "started_at": now_iso(),
        "served_model": client.config.served_model,
        "server_version": getattr(backend, "server_version", lambda: None)(),
        "healthy": getattr(backend, "health", lambda: True)(),
        "settings": {
            "structured_mode": client.config.structured_mode,
            "structured_with_reasoning": client.config.structured_with_reasoning,
            "reasoning_control": client.config.reasoning_control,
        },
    }
    items = _structured_prompts(structured_n)

    progress("structured output, reasoning off")
    report["structured_reasoning_off"] = _first_try_validity(client, items, "extract")

    reasoning_n = max(3, structured_n // 3) if reasoning_n is None else reasoning_n
    if reasoning_n:  # slow: every answer starts with a long chain of thought
        progress("structured output, reasoning on")
        report["structured_reasoning_on"] = _first_try_validity(client, items[:reasoning_n], "reason")

        progress("reasoning separation")
        reply = client.chat("Is 3599 a prime number? Answer yes or no, then one sentence why.", profile="reason")
        report["reasoning"] = {
            "reasoning_returned_separately": bool(reply.reasoning),
            "think_tags_leaked_into_answer": "<think>" in reply.text or "</think>" in reply.text,
            "answer_sample": reply.text[:200],
            "finish_reason": reply.finish_reason,
        }

    rng = random.Random(seed)
    report["long_context"] = []
    for size in context_sizes:
        progress(f"long-context recall at ~{size} tokens")
        report["long_context"].append(_needle(client, size, rng))

    report["throughput"] = []
    for level in concurrency_levels:
        progress(f"throughput at concurrency {level}")
        report["throughput"].append(_throughput(client, level))

    report["recommendations"] = recommendations(report)
    report["finished_at"] = now_iso()
    return report


def recommendations(report: dict[str, Any]) -> list[str]:
    tips: list[str] = []
    off = report.get("structured_reasoning_off", {})
    if off.get("first_try_rate") is not None and off["first_try_rate"] < 0.95:
        tips.append(
            "Structured output is unreliable on the first try: check llm.structured_mode "
            "(try 'structured_outputs' or 'guided_json' for this vLLM version)."
        )
    on = report.get("structured_reasoning_on", {})
    if on.get("n") and on["valid_after_repair"] / on["n"] < 0.9:
        tips.append(
            "JSON with reasoning on is unreliable: set llm.structured_with_reasoning: false "
            "(think first, then format in a second call)."
        )
    reasoning = report.get("reasoning", {})
    if reasoning.get("think_tags_leaked_into_answer"):
        tips.append("Reasoning leaks into answers: start vLLM with a reasoning parser (llm.server.reasoning_parser).")
    found = [c for c in report.get("long_context", []) if c.get("found")]
    lost = [c for c in report.get("long_context", []) if not c.get("found")]
    if lost:
        safe = max((c["target_tokens"] for c in found), default=0)
        tips.append(
            f"Recall failed at {[c['target_tokens'] for c in lost]} tokens; keep single prompts under "
            f"~{safe} tokens (map-reduce longer inputs)."
        )
    rates = [t for t in report.get("throughput", []) if t.get("tokens_per_s")]
    if rates:
        best = max(rates, key=lambda t: t["tokens_per_s"])
        tips.append(
            f"Best throughput {best['tokens_per_s']} tokens/s at concurrency {best['concurrency']}: "
            f"set llm.concurrency near that value."
        )
    return tips
