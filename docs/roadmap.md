# Roadmap: one tool for research in any field

Status: **proposal** (2026-11-20). Nothing here is built yet. Refine it, then build stage by stage.

## Goal

rtool should end up able to carry out the computational parts of research in any field, and to coordinate
the parts a human must do. We get there one stage at a time, and **a stage is finished only when we have
measured that it works**, not when the code exists.

Two rules follow from that:

1. Grow by **task type**, not by field. A field is a task type plus domain knowledge in a pack.
2. Build the way to **measure validity before** building the next field.

## Task types

| Task type | Examples across fields |
|---|---|
| Analyse existing data | omics, economics, ecology, social science |
| Simulate or model | systems biology, physics, epidemiology, climate |
| Benchmark methods (today's ML pipeline) | ML, bioinformatics, statistics |
| Synthesise literature | medicine, law, humanities, policy |
| Design a study a human carries out | wet lab, surveys, clinical trials, fieldwork |

Physical experiments and fieldwork cannot be run by the tool. For those it plans, analyses returned data
and writes up; a person does the physical steps.

## Validity levels

A report may claim only the level its pack has been measured at.

| Level | Meaning | How it is measured |
|---|---|---|
| L0 | Runs end to end | Run finishes without a failed stage |
| L1 | The code does what the plan says | Reviewer plus checks that results match the planned conditions and metrics |
| L2 | The statistics are sound | Code-side validators: pseudoreplication, leakage, confounding, multiple testing, seeds |
| L3 | The conclusion matches a known ground truth | Benchmark cases (below) |
| L4 | A novel finding is trustworthy | **Not claimable.** Needs outside replication by people |

The report says its level and which cases it was measured on.

## Benchmark cases

Each pack gets a folder of cases that need no human grader. One case is a brief, its data, and an expected
outcome. Four kinds:

- **Planted effect.** Synthetic data with a known effect. The tool must recover its direction and rough size.
- **Null.** Conditions that are equal by construction (for example, labels shuffled). The tool must not claim an effect. Pipeline runs are too few to measure a false-positive rate, so the statistics engine's rate is tested offline and pipeline nulls are scored as "no false claim", with an interval.
- **Replication.** A published analysis with a known result. Given the brief and the hypothesis, the tool must reach the same finding.
- **Validator unit tests.** Plain unit tests for each deterministic check (identifiers resolve, a trap is caught).

A **validity card** per pack records, for the model in use: pass rate per case kind, false-positive rate,
failure categories, cost and time. Every number is specific to one model (today Qwen3.6-35B-A3B). Changing
the model means rerunning the cases.

Each new field needs a domain expert to confirm the ground truth of its replication cases. That is the main
bottleneck per field, more than the code.

## What exists, and what is missing

Checked in the code on 2026-11-20:

| Exists | Missing for a general tool |
|---|---|
| Pack = [pack.yaml](../src/research_tool/packs/ml_dl/pack.yaml) (hints, smoke rule, packages), a code skeleton, prompt overrides | A pack cannot yet bring **validators**, **resource sources**, **literature sources** or a **reviewer rubric** |
| `fetch_resources` and the checks in [checks.py](../src/research_tool/tools/resources/checks.py) | Sources are fixed: Hugging Face, URL, torchvision, pip, manual |
| Scholarly sources in `tools/scholarly`: OpenAlex, Semantic Scholar, arXiv | No PubMed, Europe PMC or bioRxiv; none selectable per pack |
| Plan checks in [planner/checks.py](../src/research_tool/agents/planner/checks.py) (design, metrics, compute, resources) | Generic only; no per-pack hook for field-specific statistical traps |
| `rtool replay`, `rtool stats`, `rtool llm-bench`, a literature-agent evaluation (2026-10-12) | No common case format, no runner that scores L0–L3, no validity card |
| Reviewer and writer agents | Rubric is the same for every field |

## Stages

Each stage has an **exit test**. We do not start the next stage until it passes.

### Stage 0 — Pack contract and validity harness

Detailed design: [stage0_design.md](stage0_design.md).

Build:
- Pack plan checks and a `local` resource source (for case data now, and your own data later). Existing packs keep working unchanged. Literature sources and a reviewer rubric per pack are designed here but built in Stage 1, where omics is their first user.
- A case format (`benchmarks/<pack>/<case>/`), `rtool validity submit` to start case runs and `rtool validity score` to score them L0–L3.
- The validity card, generated from the scores.
- Run the **existing `generic` and `ml_dl` packs** through it first.

Exit test: validity cards for `generic` and `ml_dl` exist with a baseline for the current model, and the
runner can be rerun after any change to show what moved.

### Stage 1 — Analyse existing data

Build: tabular statistics plus an `omics` pack (scanpy, pydeseq2, gseapy). GEO and CELLxGENE as resource
sources, PubMed and Europe PMC as literature sources. Validators: pseudoreplication (cells vs donors), batch
confounding, gene and protein identifier resolution, multiple testing.

Exit test: planted effects recovered, nulls clean (no false claims, with an interval), three published
reanalyses replicated, every validator caught its planted trap.

### Stage 2 — Simulate or model

Build: a `sysbio` pack first (COBRApy for metabolism, tellurium for ODEs, BioModels as a source), then
physics and epidemiology models. Validators: conservation and mass balance, SBML validity, stiffness and
step-size checks.

Exit test: curated models reproduce their published outputs; known knockouts predicted correctly on a fixed set.

### Stage 3 — Synthesise literature (fields with no compute)

Build: a literature-only pipeline variant with no coder stage. Every claim in the report must trace to a
quote from a source (the UI already shows quotes).

Exit test: on a labelled set, claim-support accuracy and citation accuracy meet a threshold we set before running it.

### Stage 4 — Study design with a human in the loop

Build: the planner outputs a protocol (controls, power analysis, layout, materials). The user runs it and
uploads data. The analysis stages then run on that data ("bring your own data").

Exit test: protocols pass a checklist (a control for every comparison, power stated, primary outcome fixed
before data); uploaded data of a defined format flows through without manual repair.

### Stage 5 — GPU-heavy fields

Build: `protein` (ESM, deep-mutational-scanning sets), imaging, docking, then molecular dynamics. Likely a
second container image, because these packages are large.

Exit test: same harness, heavier cases. Molecular dynamics is last because its ground truth is weak.

### Stage 6 — Many users, many fields

Only after Stages 0–4 hold up. Pack packaging and sharing, per-pack access rules, usage limits. Informed by
[the full-scale architecture review](architecture.md).

## Rules that apply to every stage

- **No controlled-access data** (dbGaP, patient-level genomes). The pack must refuse it.
- **Refusal list per pack.** Pathogen enhancement and toxin or toxic-protein design are refused at the brief and plan stages.
- **Local models only**, as now. No hosted APIs.
- **Reports say what they are.** In-silico results are hypotheses, stated with their validity level.
- **A new check ships with a test that catches the trap it targets.**

## Decisions still open

1. **Who confirms ground truth** for each new field's replication cases (you, a supervisor, an outside expert)?
2. **Thresholds** for the exit tests. Proposal: false-positive rate at or below alpha, and a replication pass rate we agree before the first run, not after.
3. **Container strategy.** One big image or one per pack family. Decide at Stage 1, when `omics` first needs packages the ML image lacks.
4. **Which field after omics**, if your own research points somewhere specific.

## Not building yet

- A marketplace, user accounts beyond the current access model, or billing.
- Claims at L4.
- Cloud or hosted LLMs.
- Wet-lab automation.
