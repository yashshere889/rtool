"""Small agents that exercise every engine feature; also worked examples of the agent API.

* ``toy_source``   — LLM structured output, reads human feedback.
* ``toy_analyzer`` — in-stage gate, Slurm job + suspend/resume, checkpointing.
* ``toy_writer``   — requests a change from upstream (revision loop).
* ``toy_options`` / ``toy_chooser`` — a selection gate (pick 1 of N).
* ``toy_flaky``    — fails a configurable number of times (retry handling).
"""

from __future__ import annotations

import statistics

from pydantic import BaseModel, Field

from ..agents.base import Agent, AgentContext, FeedbackKind, StageFailed, artifact_type, register_agent
from ..hpc.jobspec import JobSpec


@artifact_type("toy_numbers")
class ToyNumbers(BaseModel):
    values: list[float]
    source: str = ""
    notes: list[str] = Field(default_factory=list)


@artifact_type("toy_stats")
class ToyStats(BaseModel):
    mean: float
    count: int
    variance: float | None = None
    job_ids: list[str] = Field(default_factory=list)


@artifact_type("toy_report")
class ToyReport(BaseModel):
    text: str
    mentions_variance: bool = False


@artifact_type("toy_options")
class ToyOptions(BaseModel):
    options: list[dict[str, str]]


@artifact_type("toy_decision")
class ToyDecision(BaseModel):
    chosen_title: str


@register_agent("toy_source")
class ToySource(Agent):
    description = "Asks the LLM for some numbers about the research topic."
    consumes = ("research_brief",)
    produces = ("toy_numbers",)

    class Config(BaseModel):
        count: int = 5

    config_model = Config

    def run(self, ctx: AgentContext) -> None:
        brief = ctx.inputs.get("research_brief")
        numbers = ctx.llm.structured(
            ToyNumbers, f"Give {self.config.count} numbers about: {brief.topic}", profile="extract"
        )
        numbers.notes = [f.text for f in ctx.feedback]  # human rejections / change requests / notes
        ctx.outputs.put("toy_numbers", numbers)


@register_agent("toy_analyzer")
class ToyAnalyzer(Agent):
    description = "Computes statistics, optionally behind a gate and a (fake) Slurm job."
    consumes = ("toy_numbers",)
    produces = ("toy_stats",)
    needs_llm = False

    class Config(BaseModel):
        use_job: bool = True
        gate: str | None = None  # in-stage gate id, e.g. "gpu_runs"
        partition: str = "gpu-l40s"

    config_model = Config

    def run(self, ctx: AgentContext) -> None:
        numbers = ctx.inputs.get("toy_numbers")
        if self.config.gate:
            decision = ctx.gate(self.config.gate, summary={"jobs": 1, "partition": self.config.partition})
            if not decision.approved:
                raise StageFailed(f"compute not approved: {decision.feedback}", needs_human=True)

        job_ids: list[str] = []
        if self.config.use_job:
            job_id = ctx.state.get("job_id")
            if job_id is None:
                spec = JobSpec(
                    name="toy-analysis",
                    partition=self.config.partition,
                    command=["true"],
                    workdir=ctx.workspace.root,
                )
                job_id = ctx.slurm.submit(spec)
                ctx.state["job_id"] = job_id
                ctx.checkpoint()  # never submit twice, even if we crash right here
            result = ctx.job_result(job_id)
            if result is None:
                ctx.suspend_until([job_id], reason="waiting for the analysis job")
            if result["state"] != "COMPLETED":
                raise StageFailed(f"analysis job {job_id} ended {result['state']}")
            job_ids.append(job_id)

        wants_variance = any(
            f.kind is FeedbackKind.CHANGE_REQUEST and "variance" in f.text.lower() for f in ctx.feedback
        )
        values = numbers.values
        variance = None
        if wants_variance:
            variance = statistics.pvariance(values) if values else 0.0
        ctx.outputs.put(
            "toy_stats",
            ToyStats(
                mean=statistics.fmean(values) if values else 0.0,
                count=len(values),
                variance=variance,
                job_ids=job_ids,
            ),
        )


@register_agent("toy_writer")
class ToyWriter(Agent):
    description = "Writes a one-line report; can demand the variance from upstream."
    consumes = ("toy_stats",)
    produces = ("toy_report",)

    class Config(BaseModel):
        require_variance: bool = False

    config_model = Config

    def run(self, ctx: AgentContext) -> None:
        stats = ctx.inputs.get("toy_stats")
        if self.config.require_variance and stats.variance is None:
            ctx.request_change(
                artifact="toy_stats",
                reason="the report needs the variance",
                required_changes=["include variance in toy_stats"],
            )
        line = ctx.llm.chat(f"Summarise: mean={stats.mean:.3f} over {stats.count} values", profile="write").text
        ctx.outputs.put("toy_report", ToyReport(text=line, mentions_variance=stats.variance is not None))


@register_agent("toy_options")
class ToyOptionsAgent(Agent):
    description = "Proposes three options for a selection gate."
    consumes = ("research_brief",)
    produces = ("toy_options",)
    needs_llm = False

    def run(self, ctx: AgentContext) -> None:
        topic = ctx.inputs.get("research_brief").topic
        ctx.outputs.put(
            "toy_options",
            ToyOptions(options=[{"title": f"{topic} — option {i}", "detail": f"detail {i}"} for i in range(3)]),
        )


@register_agent("toy_chooser")
class ToyChooser(Agent):
    description = "Consumes the option a human selected."
    consumes = ("toy_choice",)
    produces = ("toy_decision",)
    needs_llm = False

    def run(self, ctx: AgentContext) -> None:
        selected = ctx.inputs.get("toy_choice")
        ctx.outputs.put("toy_decision", ToyDecision(chosen_title=selected.item["title"]))


@register_agent("toy_flaky")
class ToyFlaky(Agent):
    description = "Fails the first N attempts, then succeeds."
    consumes = ("research_brief",)
    produces = ("toy_numbers",)
    needs_llm = False

    class Config(BaseModel):
        fail_times: int = 1

    config_model = Config

    def run(self, ctx: AgentContext) -> None:
        if ctx.attempt <= self.config.fail_times:
            raise RuntimeError(f"simulated failure on attempt {ctx.attempt}")
        ctx.outputs.put("toy_numbers", ToyNumbers(values=[1.0, 2.0], source="flaky"))
