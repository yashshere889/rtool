# Adding an agent

An agent is one class: it declares which artifacts it reads and writes, and
`run(ctx)` does the work. You add a stage for it to a pipeline YAML and the
engine does the rest — scheduling, persistence, gates, retries, staleness,
Slurm episodes. No engine code changes.

This walk-through adds a **novelty checker**: after you pick a hypothesis it
asks the model which paper in the corpus is closest, and records the verdict.
(The code blocks below are run by `tests/integration/test_adding_an_agent.py`,
so they stay correct.)

## 1. The artifact it produces

Artifacts are Pydantic models registered under a name. Keep them small and
explicit; whatever later stages or humans need goes here.

```python
# file: my_agents/novelty.py
from pydantic import BaseModel, Field

from research_tool.agents.base import Agent, AgentContext, StageFailed, artifact_type, register_agent


@artifact_type("novelty_report")
class NoveltyReport(BaseModel):
    hypothesis: str
    closest_paper_id: str
    closest_title: str
    overlap: str
    score: int  # 1 = already done, 5 = clearly new
```

## 2. What the model fills in

Ask the model for as little as possible, in a narrow schema. A 12B model is
reliable on small, checkable questions; field descriptions are part of the
prompt it sees.

```python
# file: my_agents/novelty.py
class NoveltyVerdict(BaseModel):
    closest_paper_id: str = Field(description="Id of the most similar paper, copied from the list")
    overlap: str = Field(description="One sentence: what that paper already did")
    score: int = Field(ge=1, le=5, description="1 = already done, 5 = clearly new")
```

## 3. The agent

```python
# file: my_agents/novelty.py
@register_agent("novelty_checker")
class NoveltyChecker(Agent):
    description = "Finds the corpus paper closest to the chosen hypothesis."
    consumes = ("selected_hypothesis", "corpus")
    produces = ("novelty_report",)

    class Config(BaseModel):
        max_papers: int = 40  # set per stage in the pipeline YAML

    config_model = Config

    def run(self, ctx: AgentContext) -> None:
        hypothesis = ctx.inputs.get("selected_hypothesis").item["statement"]
        papers = ctx.inputs.get("corpus").included()[: self.config.max_papers]
        listing = "\n".join(f"- {p.id}: {p.title}" for p in papers)
        prompt = (
            f"Hypothesis: {hypothesis}\n\nPapers:\n{listing}\n\n"
            "Which paper is closest to this hypothesis, what did it already do, and how new is the hypothesis?"
        )
        verdict = ctx.llm.structured(NoveltyVerdict, prompt, profile="judge")
        closest = next((p for p in papers if p.id == verdict.closest_paper_id), None)
        if closest is None:  # check the model's answer in code, never trust it blindly
            raise StageFailed(f"the model named a paper that is not in the list: {verdict.closest_paper_id!r}")
        ctx.outputs.put(
            "novelty_report",
            NoveltyReport(
                hypothesis=hypothesis,
                closest_paper_id=closest.id,
                closest_title=closest.title,
                overlap=verdict.overlap,
                score=verdict.score,
            ),
        )
```

What `ctx` gives you:

| | |
|---|---|
| `ctx.inputs.get(name)` | the exact input versions this attempt reads |
| `ctx.outputs.put(name, model, meta=...)` | stage an output; committed only if `run` returns normally. `meta={"review": "<run-relative path>"}` shows a file at the gate |
| `ctx.llm` | `chat`, `structured`, `structured_many`, `parallel` — with a profile name |
| `ctx.state` + `ctx.checkpoint()` | anything to keep across a crash, a gate or a Slurm wait |
| `ctx.workspace` | a path-jailed folder for this stage's files |
| `ctx.gate(id, summary)` | ask a human mid-stage (before something costly) |
| `ctx.slurm.submit(spec)` + `ctx.suspend_until([job])` | run jobs; `run` is called again when they end, `ctx.job_result(id)` |
| `ctx.request_change(artifact=..., reason=...)` | ask the producer of an input to redo its work |
| `ctx.feedback` | human rejections and change requests addressed to this stage |
| `ctx.resume_without_llm()` | the next resumption may start on a CPU node |
| `ctx.tools[...]` | injected tools (fakes in tests, site-specific clients) |
| `raise StageFailed(msg, needs_human=True)` | stop and ask a person to fix something |

Prompts usually live next to the agent as Jinja templates
(`prompts/check.j2`, rendered with `PromptLibrary.for_agent(__package__, "novelty_checker")`)
so domain packs can override them and their hash is recorded; an inline string is fine to start.

## 4. Make rtool import it

Inside this repository: add the module to `BUILTIN_AGENT_MODULES` in
`src/research_tool/core/registry.py`. In your own package: declare an entry
point and install the package next to rtool —

```toml
[project.entry-points."research_tool.agents"]
novelty = "my_agents.novelty"
```

`rtool agents` lists everything registered.

## 5. Put it in a pipeline

Copy `src/research_tool/pipelines/research.yaml` and add the stage. Its place
in the order comes from what it reads: it runs once `selected_hypothesis`
exists, i.e. after the hypothesis gate.

```yaml
  - id: novelty
    agent: novelty_checker
    config: {max_papers: 60}
    gate: novelty_review       # optional: declare it under `gates:` to look at the report first
```

```bash
rtool pipelines --check my_research.yaml     # resolves agents, artifacts and order
rtool new sleep --topic "..." --pipeline my_research.yaml
```

## 6. Test it without a GPU

`MockLLM` answers by rules and fills unmatched schemas with valid examples;
`make_run` creates a run with a research brief, and `Engine(...).advance()` is
one episode.

```python
# file: tests/test_novelty.py
from research_tool.core.artifacts import ArtifactStore, Producer
from research_tool.core.engine import Engine, Outcome
from research_tool.core.services import Services
from research_tool.llm.client import LLMClient
from research_tool.llm.mock_backend import MockLLM
from research_tool.schemas.common import SelectedItem
from research_tool.schemas.literature import Corpus, PaperRecord
from research_tool.testing.harness import make_run


def test_novelty_checker(runs_root):
    pipeline = {
        "name": "novelty-only",
        "initial_artifacts": ["research_brief", "selected_hypothesis", "corpus"],
        "stages": [{"id": "novelty", "agent": "novelty_checker"}],
    }
    run = make_run(runs_root, pipeline)
    store = ArtifactStore(run.artifacts_dir)
    me = Producer(stage="test", agent="test")
    store.put("selected_hypothesis", SelectedItem(index=0, item={"statement": "Sound boosts spindles"}), producer=me)
    papers = [PaperRecord(id="doi:1", title="Closed-loop stimulation of slow waves", status="extracted")]
    corpus = Corpus(topic="t", queries=[], papers=papers, kb_path="", kb_sha256="", bibtex_path="")
    store.put("corpus", corpus, producer=me)

    llm = MockLLM().add("Which paper is closest", {"closest_paper_id": "doi:1", "overlap": "slow waves", "score": 3})
    result = Engine(run, Services(llm=LLMClient(llm, run.config.llm))).advance()

    assert result.outcome is Outcome.DONE
    assert store.get("novelty_report").closest_title.startswith("Closed-loop")
```

## Checklist

- Read and write only artifacts; never import another agent or refer to stage ids.
- Narrow LLM schemas; check every answer in code (ids exist, quotes match, numbers come from data).
- Long or costly work: keep progress in `ctx.state` and `ctx.checkpoint()` so a resumed stage skips what is done.
- Before spending GPU hours, ask: `ctx.gate("my_gate", summary={...})`.
- Set `needs_llm = False` (or subclass `Task`) if the stage never calls the model: it then runs in cheap CPU episodes.
