# Architecture

rtool turns a research topic into a reviewed report with six AI agents, running on
Barkla with a local LLM (Nemotron Nano 12B v2). This page explains how the pieces
fit together; [adding_an_agent.md](adding_an_agent.md) shows how to extend it and
[barkla_runbook.md](barkla_runbook.md) how it runs on the cluster.

## The pipeline

```
literature ─▶[literature_review]─▶ hypotheses ─▶[hypothesis_selection]─▶ plan ─▶[experiment_plan]─▶ fetch
   ─▶ code: write ─▶ smoke test ─▶[gpu_runs]─▶ Slurm runs ─▶ analysis ─▶ report ─▶ review ─▶[final_report]
            ▲______________________ change requests (bounded) ______________________│
```

| stage | agent | reads | writes |
|---|---|---|---|
| literature | `literature` | research_brief | corpus, knowledge_map |
| hypotheses | `hypothesis` | brief, corpus, knowledge_map | hypothesis_set (+ selected_hypothesis from the gate) |
| plan | `planner` | brief, selected_hypothesis, corpus, knowledge_map | experiment_plan |
| fetch | `fetch_resources` (task) | experiment_plan | resource_manifest |
| code | `coder` | brief, experiment_plan, resource_manifest | code_bundle, experiment_results |
| report | `writer` | brief, corpus, knowledge_map, plan, results, code_bundle | report |
| review | `reviewer` | the same + report | review, final_report |

The pipeline is YAML (`src/research_tool/pipelines/research.yaml`): stages, the
agent each runs, per-stage config and gates. Dependencies come from what each
agent reads and writes, so reordering or inserting stages needs no code.

## Core ideas

**Artifacts, never direct calls.** Agents exchange typed, versioned artifacts
(Pydantic models registered with `@artifact_type`). Each version records the
stage, agent, attempt and the exact input versions it came from. When an input
changes, every stage that read it becomes *stale* and only those re-run.

**The engine runs episodes.** `Engine.advance()` takes the run's lease, applies
queued human actions, runs every ready stage, persists everything atomically and
stops when the run needs a person (a gate), a Slurm job, the LLM, or more time.
Kill it at any point and the next episode resumes: a stage that was running is
re-entered with its saved `ctx.state`; a commit that was half done is finished.

**Gates.** After a stage (`gate:` in the YAML) or inside one (`ctx.gate(...)`,
e.g. the Coder's `gpu_runs` before spending GPU hours). Humans answer with
`rtool approve | reject -f "…" | select <n>`; any gate can be set to `auto`.

**Change requests.** A later stage can ask the producer of one of its inputs
to redo its work (`ctx.request_change(artifact=…)`): the Writer when planned
conditions have no results, the Reviewer for major problems with the experiment.
The engine marks the target stale with the request as feedback, re-runs it,
then the requester. Rounds are capped (`revisions.max_rounds`); past the cap the
`revision_limit` gate asks a human.

**Nothing is resident.** The CLI only writes small action files and submits
Slurm jobs; all work happens in episode jobs (next section).

## Episodes on Slurm

An *episode* is a Slurm job running `rtool episode <run>`. When it stops it
decides what comes next:

| outcome | next |
|---|---|
| a stage needs the LLM, this is a CPU job | a GPU episode, on the GPU Slurm expects to start soonest (`slurm.episode.gpu_choices`, priced with `sbatch --test-only`) |
| waiting for experiment jobs | an episode that starts when they end (`--dependency=afterany`); it starts on a CPU node and moves to a GPU only if needed |
| a gate is open | nothing; `rtool approve/reject/select` submits a (CPU) episode |
| time budget used | another episode right away |
| done / failed | nothing |

CPU episodes are cheap and start quickly, so human decisions and bookkeeping
happen there. GPU episodes start vLLM lazily, only when a stage calls the model.
A stage can declare that its next resumption needs no model
(`ctx.resume_without_llm()`): the Coder does this for submitting and collecting
jobs, so a run never waits in the GPU queue just to do bookkeeping; if such a
step needs the model after all, touching `ctx.llm` hands the stage to a GPU
episode with its state intact.

Episodes are requeued if preempted (low-priority partitions): `TERM` makes the
episode stop cleanly within Slurm's grace period, and the requeued job — same
id — reclaims its own lease and resumes. One episode per run at a time (a lease
file); a time-limit warning (`USR1` 10 min before) queues the continuation.

## The LLM layer

Agents name *profiles* (`extract`, `reason`, `code`, `write`, `judge`), never a
model. The client gives them:

- `structured(Model, prompt)` — JSON constrained by the model's schema (vLLM
  structured outputs), validated with Pydantic; invalid output is shown back
  with the error and retried. A reply that ran out of tokens (greedy decoding
  repeating itself) is retried briefly re-prompted and sampled instead.
- reasoning on/off via Nemotron's `/think` / `/no_think`; reasoning is split
  from the answer. With reasoning on, JSON is produced in a second call.
- `parallel(...)` — many requests at once (32 on Barkla), batched by vLLM.

Every prompt is a Jinja template next to its agent; its hash is stored with the
artifact it produced. Domain packs can override any prompt.

## Why a 12B model can be trusted here

The model is good but small, so every step is narrow and checked by code:

| risk | guard |
|---|---|
| invented findings | every extracted claim carries a quote that must match the paper text (fuzzy, ≥ 0.85) |
| invented citations | hypotheses cite verified quotes; reports may only cite corpus keys |
| invented numbers | reports write results as placeholders (`{{mean:acc:new}}`) filled from the results; any other number must appear in the material the section was written from |
| wrong statistics | all statistics are computed deterministically (t-tests, exact rank tests, CIs, corrections) |
| broken code | static checks, CPU smoke tests, a failure playbook, bounded LLM repairs, then a human |
| runaway cost | a human approves every GPU spend; budgets for GPU-hours, jobs and LLM calls |
| overconfident review | three independent reviews; scores are medians; an issue counts only if most raise it |

## Domain packs

`src/research_tool/packs/<name>/`: `pack.yaml` (hints, what a smoke run is,
usual packages), `templates/experiment.py.j2` (the Coder's skeleton) and
optional prompt overrides `prompts/<agent>/*.j2`. Built in: `generic` and
`ml_dl` (PyTorch). Choose with `rtool new … --pack ml_dl`.

## A run on disk

```
<runs_root>/<project>/<run_id>/
  run.json             the pipeline and config snapshot
  state/               run and per-stage state (status, attempts, saved agent state, gates, usage)
  actions/             human decisions waiting for the next episode (and the processed ones)
  artifacts/<name>/    every version of every artifact (JSON with provenance)
  events.jsonl         everything that happened, append-only
  jobs/                Slurm bookkeeping, vLLM logs
  work/<stage>/        each stage's workspace: knowledge base, code, runs, results, report, review
```

`rtool archive <run>` packs everything except rebuildable files into one
`.tar.gz` for the backed-up `data` area.
