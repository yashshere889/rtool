# Stage 0 design: measuring whether rtool's results are right

Status (2026-11-21): PRs 1–3 are **built** and committed. PR 4, the baseline on Barkla, is written up in
[barkla_runbook.md](barkla_runbook.md#validity-baseline-stage-0-pr-4) and **not run**, so the exit test
below has not been met. Part of [the roadmap](roadmap.md); what was built is in [build_log.md](build_log.md).

## What Stage 0 delivers

A way to say, with numbers, how often rtool reaches the right conclusion, using today's packs
(`generic`, `ml_dl`) on today's model. Every later stage reuses it: a new field counts as done
only when its pack passes the same harness.

**Exit test.** Validity cards for `generic` and `ml_dl` exist for the current model, produced from a
clean commit. The harness can be rerun after any change to show what moved.

## What it measures, and what it does not

Cases start at the **plan** stage with a fixed hypothesis and fixed conditions. So Stage 0 measures
**execution validity**: given a clear question, does rtool plan, code, run, analyse and report it
correctly?

It does **not** measure:
- Literature search quality. `python -m research_tool.agents.literature.measure` already does that.
- Hypothesis quality.
- How well the planner designs an experiment from an open question.

One `open` case per pack runs the whole pipeline, but it is scored only for finishing (L0) and code
consistency (L1).

## How it fits together

```
benchmarks/<pack>/<case>/case.yaml         the case: brief, hypothesis, data, expected outcome
        │
rtool validity submit <pack>                (runs on the login node: seconds, then exits)
        │  for each case × variant × repeat:
        │    generate the case data (stdlib Python) ──▶ local data folder, content-hashed
        │    create a run: pipeline derived from research.yaml, starting at `plan`
        │    seed research_brief, selected_hypothesis, an empty corpus and knowledge map
        │    config overlay: every gate auto, hard budgets
        │    submit its first episode
        ▼
ordinary episodes on Slurm (plan → fetch → code → report → review), no human
        │
rtool validity score <pack>                 (idempotent: scores runs that are at rest)
        ▼
validity card (JSON + Markdown): levels per case, false claims, power, failures, cost, provenance
```

Nothing stays resident. That follows the architecture rule: no long processes on login nodes. Both
commands return in seconds. `score` can be run as often as you like and only reads files on disk.

## 1. Case format

```
benchmarks/
  ml_dl/
    label-noise-hurts/
      case.yaml
      reference.py      optional: a hand-written version that confirms the expected effect at this budget
    xor-needs-hidden-layer/
      case.yaml
      generate.py       optional: writes the case's data; stdlib only, deterministic
```

`case.yaml`:

```yaml
id: xor-needs-hidden-layer
kind: planted                    # planted | null | replication | open
pack: ml_dl
start_at: plan                   # plan (default) | literature (open cases only)

brief:                           # becomes research_brief; the pinned fields are enforced by plan checks
  topic: Do hidden layers help on a non-linearly separable problem?
  seeds: 8                       # at least 6: see rule 6
  max_minutes_per_run: 5
  datasets: [xor_blobs]
  conditions:                    # NEW brief field: fixed condition names (see section 4)
    - {name: linear, baseline: true, description: logistic regression on the two inputs}
    - {name: mlp, description: one hidden layer of 32 ReLU units}
  primary_metric: {name: test_accuracy, higher_is_better: true}   # NEW brief field
  constraints: [CPU only]

hypothesis:                      # becomes selected_hypothesis (fields of schemas.hypothesis.Hypothesis)
  statement: An MLP with one hidden layer reaches higher test accuracy than logistic regression on xor_blobs.
  null_hypothesis: Test accuracy does not differ between the two models.
  independent_variables: [model]
  dependent_variables: [test_accuracy]
  success_criteria: mlp's mean test_accuracy is higher than linear's, and the paired t-test gives p < 0.05.
  compute: cpu_only

data:                            # local datasets this case provides (section 3)
  - name: xor_blobs
    generator: generate.py
    args: {n: 4000, noise: 0.3, seed: 7}
    description: Two inputs, a binary label from XOR of their signs; train.csv and test.csv.

expect:
  significant: true
  direction: better              # better | worse (in the metric's preferred direction); omit for nulls
  difference: {low: 0.25, high: 0.50}   # optional: band for the estimated difference, fixed by reference.py
  outcome: [supported]           # what final_report.hypothesis_outcome may say

variants:                        # optional: each variant is a separate run (gives a power curve)
  - {name: strong, data_args: {noise: 0.3}}
  - {name: weak, data_args: {noise: 1.2}, expect: {difference: {low: 0.02, high: 0.20}}}

repeats: 3
```

The runner fills the fields a `Hypothesis` needs but a case doesn't care about: `id` comes from the
case id, `evidence` is `[]`, and the title comes from the statement.

### Case kinds

| Kind | Truth comes from | `expect` |
|---|---|---|
| planted | the generator or the simulation: the effect exists by construction | significant, direction, optional difference band |
| null | exchangeability: the conditions are equal in distribution under the seed | `significant: false`, outcome `[refuted, inconclusive]` |
| replication | a known value (analytic, or published and confirmed by `reference.py`) | `value: {condition, low, high}` on the primary metric's mean |
| open | nothing: topic only, full pipeline | none; scored L0–L1 only |

### Rules for case authors

1. **A null must be exchangeable under the seed.** The two conditions must be the same process in
   distribution, with the seed driving all the randomness. If not, a "false positive" may be a real
   small effect.
   - Good: train on seeded random half A vs half B; the same simulation with two RNG algorithms.
   - Bad: "with a noise feature vs without it". Extra noise features usually hurt a little, so the true
     effect is not zero.
2. **Confirm the effect before the case counts.** A planted or replication case needs a `reference.py`
   (or an analytic value) showing the effect at the case's budget: same seeds, same minutes. Otherwise
   a failure may be the case's fault, not rtool's.
3. **Write `success_criteria` as a test on the primary metric**, naming the conditions. The Writer's
   `hypothesis_outcome` is computed from it, so a vague criterion scores the wording, not the pipeline.
4. **Stage 0 cases are seed-replicated computations** (simulations, training runs). Analysing one
   fixed dataset needs sample-level statistics (tests over data points, not seeds). That is a Stage 1
   design problem, so no Stage 0 case analyses a fixed table.
5. **Cases must be CPU-feasible** for the Stage 0 baseline, so experiments need no GPU time (section 9).
6. **Use at least 6 seeds.** The planner may pick the exact Wilcoxon signed-rank test, one of the
   schema's own examples. With 5 seeds its smallest two-sided p-value is 2/32 = 0.0625, so it can
   never be significant, and a real effect would score `wrong_analysis`. With 6 seeds the smallest is
   2/64 ≈ 0.031.

## 2. Running a case

### Derived pipeline

A case does not need its own pipeline file. `start_at: plan` means: take `research.yaml`, drop every
stage before `plan`, and set `initial_artifacts` to whatever the remaining stages consume but no
remaining stage produces. That gives `research_brief`, `selected_hypothesis`, `corpus` and
`knowledge_map`. `resolve_pipeline` already rejects a spec where a stage consumes something nobody
provides ([pipeline.py](../src/research_tool/core/pipeline.py), the `every_artifact` / `consumes`
check). So a wrong derivation fails at `submit`, not in an episode.

New code: one function, `pipeline_from(spec, start_at) -> PipelineSpec`, next to `find_pipeline`.

### Seeded artifacts

| Artifact | Seeded as |
|---|---|
| `research_brief` | from `case.brief` |
| `selected_hypothesis` | `SelectedItem(index=0, item=<Hypothesis from case.hypothesis>)` |
| `corpus` | `Corpus(topic, queries=[], papers=[], kb_path="seeded/no-kb.sqlite", ...)`. The path must not exist: [`open_corpus_kb`](../src/research_tool/agents/common.py) only checks `.exists()`, and an empty `kb_path` would resolve to the run folder, which does exist |
| `knowledge_map` | `KnowledgeMap(topic, overview="")` |

All are stored with `Producer(stage="user", agent="rtool validity")`, as `create_run` does for the brief.

**Risk:** the Writer and Reviewer had never run with an empty corpus. PR 1 runs it with the demo model
(`tests/integration/test_validity.py`): the run finishes and scores L3. The demo model is scripted,
though. The first real run will show whether the real model's complaints about missing related work
turn into change requests. If they do, the overlay sets the Reviewer's change requests to report-only
for validity runs.

### Config overlay

Written to `validity-overlay.yaml` in the run folder. It is merged into the run's config and also added
last to `slurm.episode.config_files`, as [`replay_run`](../src/research_tool/service/runs.py) does.
Episodes re-read those files over the run's snapshot, so a file such as `barkla.yaml` can never turn a
gate back to "ask".

```yaml
gates:
  default: auto
  modes: {experiment_plan: auto, final_report: auto, gpu_runs: auto, manual_resources: auto, revision_limit: auto}
budgets:
  run: {max_gpu_hours: 0, max_llm_calls: 3000, max_jobs: 40}   # experiments CPU-only in the baseline
```

`modes` names every gate explicitly: the pipeline's own, the in-stage ones, and any the config already
sets. A per-gate "ask" in a config file would otherwise beat `default: auto`.

I checked that in-stage gates go through `config.gates.mode()` (`core/context.py:183`), and so does
`revision_limit` (`core/engine.py:548`). So no validity run can stall waiting for a person. Because
`gpu_runs` is auto, the overlay's budgets are the only thing standing between a case and GPU spend.
`submit` therefore refuses an overlay with no `max_gpu_hours`.

A budget of 0 is a real limit, not "unlimited": `budget_issue` in `planner/checks.py` tests
`budget is None`, and `core/budgets.py` tests `limit is not None`. A plan with GPU runs therefore gets a
budget issue and is re-planned.

To verify in PR 1:
- The planner then settles on a CPU partition, rather than failing the budget check on every try.
- The LLM episodes' own GPU time is not counted in this budget.

### Commands

- `rtool validity submit <pack> [--case ID ...] [--repeats N]` creates and starts the runs. Project:
  `validity-<pack>`. Run id: `<case>-<variant>-r<k>-<commit7>`. It skips runs that already exist, so
  rerunning it fills gaps.
- `rtool validity score <pack> [--commit SHA]` scores every run of that commit that is at rest (done,
  failed, or needing a person). It lists runs still going, and writes the card.

Named `validity`, not `bench`, because `rtool llm-bench` already measures LLM speed.

## 3. Local data: the `local` resource source

Planted data has no way to reach the Coder today. The sources are `huggingface | url | torchvision |
pip | manual`, and a `manual` resource is flagged as a plan issue and waits at the `manual_resources`
gate. Stage 4's bring-your-own-data needs the same thing, so this is built once, here.

- **Where data lives:** `submit` runs each case's `generate.py` with `python -I` into a temporary
  folder, hashes the files, and moves them to `<data root>/<name>-<hash12>/`. So identical data is
  shared, and changed data never overwrites a folder an earlier run used.
- **How a run finds it:** a new config map, `resources.local: {<name>: {path, description}}`, written
  into the run's overlay by `submit`. The model only ever sees and writes the plain name (`xor_blobs`);
  the map turns it into the hashed folder. A model copying a 12-character hash would fail for reasons
  that have nothing to do with rtool.
- **Schema:** `ResourceSource` gains `"local"`. The identifier must be a name in that map. It comes from
  the model, so this is a trust boundary: no paths, no `..`, unknown names rejected.
- **Check** ([checks.py](../src/research_tool/tools/resources/checks.py)): `local` is verified if the
  name is in the map and its folder exists, with its size from `tree_size`.
- **Fetch** ([fetch.py](../src/research_tool/tools/resources/fetch.py)): copy it into
  `<resources_root>/local/<name>/` and record the content hash in the manifest. Experiments already
  bind the resources root, so they need no new bind.
- **Planner prompt:** one line listing the map's datasets (name, description, files), the same way
  cached models are listed today (`cached_models` in the planner). `brief.datasets` already pins
  which ones may be used (`brief_issues`).

**Decision:** the `ResourceChoice.source` `Literal` is part of the structured-output schema the model
must follow. Per-pack sources would mean building that schema per pack. Instead, the `Literal` stays
one superset for all packs, and a pack lists the sources it allows (`resource_sources` in
`pack.yaml`, default: all). A plan check rejects anything else. Simpler, and the schema stays fixed.

## 4. Pinned design in the brief

The scorer has to find "the" comparison that decides the case. The planner names conditions and
metrics freely (`noisy_labels`, `acc`, `test_err`...), so the case pins them through two new brief
fields. They are enforced the way PRs 5 and 6 enforce `seeds` and `datasets`: in `brief_issues`, with
prompt lines from `ResearchBrief.requirements()`.

- `conditions: [{name, baseline, description}]`. The plan must contain these conditions with these
  names, and exactly one of them is the baseline. Ablations may be added.
- `primary_metric: {name, higher_is_better}`. The plan's primary metric must have this name and direction.

These are also useful outside validity runs: a researcher who pre-registers the arms and the primary
outcome gets them enforced. The cost is that validity cases constrain the planner more than a real
user does. That is accepted for Stage 0 and stated on the card.

## 5. Pack contract

Built now:

| Hook | In `pack.yaml` | Called |
|---|---|---|
| Plan checks | `plan_checks: ["research_tool.packs.ml_dl.checks:held_out_issues"]` | On the finished plan, in the planner and in `plan_issues` (hand edits). Signature `(plan: ExperimentPlan, brief: ResearchBrief) -> list[str]`, like the existing `*_issues` functions. Results go to `plan.issues` |
| Allowed resource sources | `resource_sources: [huggingface, torchvision, pip, local]` | By a plan check (section 3) |

The first pack check, which proves the hook, goes in `ml_dl`. `held_out_issues` flags a plan whose
dataset resources all name `split: train`, with no test or validation split and no split in code.
That is the plan-level shadow of the Reviewer's `held_out` audit question. It should rarely fire
falsely: plans that leave `split` empty are not flagged.

Found while designing this, and worth fixing for every user: a plan whose named test cannot reach
alpha with its number of seeds (rule 6) is accepted today. That is a generic plan check, not a pack
one, and goes in PR 2.

First version: a pack check's issues are recorded and shown, not fed back into the planner's retries.
In auto mode that means an L2 failure, which is what we want to measure. Feed them back once the cards
show they fire often.

Designed now, built in Stage 1, when omics is their first user:

| Hook | Interface |
|---|---|
| Literature sources | `literature_sources: [openalex, semantic_scholar, arxiv, pubmed]` in `pack.yaml`. Each name is a class with the interface the existing ones share in [sources.py](../src/research_tool/tools/scholarly/sources.py). The Literature agent uses the pack's list instead of its fixed one |
| Reviewer rubric | `prompts/reviewer/rubric.j2`: the existing per-pack prompt override, plus a list of required checks the scorer reads back from `review.checks` |

## 6. Scoring

Deterministic. It reads only artifacts and state on disk; no model is called. A run's level is the
highest level whose rule passes, together with every level below it.

| Level | Reads | Passes when |
|---|---|---|
| L0 runs | `state/` | every stage DONE, and no human action was applied (the run state's `processed_actions` is empty: gates were auto and nobody retried or edited). Stage failure counts are recorded, not failed |
| L1 code consistent | `experiment_results`, `code_bundle`, `experiment_plan` | `complete` is true; every pinned condition has every seed `ok`; `code_fingerprint == code_bundle.fingerprint`; no `warnings` on either (fake-experiment signs) |
| L2 analysis sound | `experiment_plan`, `experiment_results` | `plan.issues` is empty (including pack checks); seeds match the brief; a `confirmatory` comparison exists for the pinned condition against the pinned baseline on the pinned primary metric, with a p-value |
| L3 right answer | `experiment_results`, `final_report` | **Analysis:** that comparison's `significant` and `better` match `expect`, its `difference` is inside the band if given, and for replication the condition's mean is inside `value`. **Report:** `final_report.hypothesis_outcome` is in `expect.outcome` |

The Reviewer's `code_audit` (problem and unconfirmed counts) is shown next to L1 but never decides it.
It is an LLM judge, and the scorer stays deterministic.

Two things checked so that no level reads zero by construction:
- `final_report` is the Writer's report copied with the Reviewer's edits (`model_copy` in the reviewer
  agent), so `hypothesis_outcome` carries over.
- torchvision and pip resources count as verified in `checks.py`, so they do not add plan issues.

PR 1 still confirms on existing run artifacts that `plan.issues` is normally empty. Otherwise L2 would
fail for reasons unrelated to soundness.

**Failure categories** (counted on the card):

| Category | Meaning |
|---|---|
| `stage_failed:<stage>` | an L0 failure, with the error's first line |
| `stalled_at_gate:<gate>`, `human_intervened` | L0 failures: the run waited for a person, or a person acted on it |
| `incomplete_runs`, `missing_condition`, `fake_experiment`, `stale_results` | L1 failures (`missing_condition`: the plan left out a pinned condition) |
| `plan_issue`, `missing_comparison` | L2 failures |
| `wrong_analysis` | the statistics disagree with the truth (code or design wrong) |
| `misread_results` | the statistics are right but the report's outcome is wrong (the Writer) |
| `false_claim` | the report says `supported` on a null case. Counted separately: it is the worst error |

The scorer is tested on fixture runs made with the demo model (`testing/demo_run.py`): one for each
level, plus a null case where the report claims `supported`.

## 7. Statistics: what can and cannot be measured

"False positives at or below alpha" cannot be measured from pipeline runs we can afford. With 10 null
runs and 0 false positives, the 95% Wilson interval on the true rate still reaches about 28%. So
there are two separate measurements:

1. **The statistics engine, offline.** A unit test, `tests/unit/test_comparison_calibration.py`,
   draws thousands of simulated `RunRecord` sets from known distributions and runs them through
   [agents/coder/stats.py](../src/research_tool/agents/coder/stats.py). It checks three things: the
   false-positive rate is at or below alpha (near alpha for the t-tests; the exact rank tests are
   conservative at small n, and Wilcoxon signed-rank with 5 paired seeds has a false-positive rate
   and power of 0);
   Holm keeps the family-wise rate at or below alpha; the t-tests' power matches theory at d = 1. It
   is pure Python with no model, runs in seconds, and goes in CI.
2. **The pipeline, on cases.** Null cases are scored on "no false claim". The card gives k / N with a
   Wilson interval and says plainly how wide it is.

**What the calibration test measured** (8 paired seeds unless noted, 600–2,000 simulated experiments each):

| Analysis | Measured | Expected |
|---|---|---|
| Paired t-test on a null | false-positive rate 0.048 | 0.05 |
| Paired t-test at d_z = 1 | power 0.681 | 0.681 (noncentral t) |
| Wilcoxon signed-rank on a null | 0.034 | at most 0.05 (exact test) |
| Mann–Whitney U on a null (6 seeds) | 0.015 | at most 0.05 |
| Three comparisons with Holm | family-wise rate 0.046–0.054 (three batches of 4,000); 0.149 uncorrected | at most 0.05 |

**Found:** when one run fails, `aggregate` drops the pairing for that comparison and uses a Welch
test on all remaining runs. Seeded runs share variation across conditions, so this throws most of it
away. Power at d_z = 1 falls from 0.70 to 0.44, while pairing on the 7 seeds left would keep 0.61
(false positives 0.01 instead of 0.05). Fixing it means pairing on the shared seeds; that is a change
to the analysis, not to the harness, so it is not in PR 1.

Repeats are informative: the model profiles sample (temperature 0.6–1.0 in `core/config.py`), so the
same case can go differently on a second run. Generator variants (effect size, noise, n) give a power
curve per pack. That is more useful than repeating one variant many times.

## 8. The validity card

Written by `score` as `validity/<pack>/<commit7>.json` and `.md` under the runs root. The baseline
card is copied into the repo as `benchmarks/<pack>/CARD.md`.

- **Headline:** runs at each level (L0–L3) per case kind, with Wilson intervals.
- **False claims** on null cases: k / N with an interval.
- **Power curve:** L3 pass rate per variant, for cases with variants.
- **Failure categories**, with counts and the run ids to open.
- **Cost:** LLM calls, tokens, wall time per stage, GPU episode hours. All from `stats_for_run_dir`
  ([core/stats.py](../src/research_tool/core/stats.py)).
- **Provenance:** rtool commit and dirty flag; model id and vLLM version from the config and events;
  prompt template hashes (already stored with each artifact); the case-set hash (case files plus data
  hashes).
- **Limits:** what was not measured (section "What it measures").

`submit` refuses to run from a dirty tree, and `score` marks any run whose recorded commit is dirty.
Barkla currently runs hand-patched code that matches no commit. A card from that would be
unattributable, so the baseline needs a clean deploy first.

## 9. The first case set

All CPU-only, so experiments need no GPU time. The LLM episodes still need GPUs. Each expected value is
confirmed by `reference.py` or an analytic formula before the case counts (rule 2).

| Pack | Case | Kind | Truth | What it exercises |
|---|---|---|---|---|
| generic | `mm1-time-in-system` | replication | M/M/1 with λ = 0.8, μ = 1: mean time in system is 1/(μ − λ) = 5 | simulation, a value check |
| generic | `two-servers-shorter-wait` | planted | M/M/2 vs M/M/1 at the same λ: much lower time in system | direction on a lower-is-better metric |
| generic | `rng-choice-null` | null | the same M/M/1 run with PCG64 vs MT19937 | no false claim; variants with 3 values of λ |
| ml_dl | `xor-needs-hidden-layer` | planted | XOR data is not linearly separable | the `local` source; variants by noise |
| ml_dl | `label-noise-hurts` | planted | 40% vs 0% training label noise on an MNIST subset lowers test accuracy | torchvision fetch; direction `worse` |
| ml_dl | `random-halves-null` | null | the same MLP on seeded random halves A vs B of the training subset | no false claim; variants by subset size |
| ml_dl | `softmax-mnist-accuracy` | replication | linear softmax on MNIST, test accuracy in a band fixed by `reference.py` | evaluation on the test split |
| ml_dl | `open-topic` | open | none | the whole pipeline finishes (L0–L1) |

With 3 repeats and the listed variants, that is about 40 runs. Cost per run is unknown until the first
one is measured. So the baseline starts with one run per case, and repeats follow once the cost is known.

## 10. Work breakdown

| PR | Contents | Runs where |
|---|---|---|
| 1 | Case format and loader; `pipeline_from`; seeding; overlay; scorer; scorer fixtures with the demo model; the offline stats calibration test. The pinned conditions and primary metric reach the planner as requirement lines in the brief, not yet enforced; runs of cases with their own data are refused until PR 2 | Mac / CI only. **Built:** `src/research_tool/validity/`, `pipeline_from` in `core/pipeline.py` |
| 2 | `local` source (schema, check, fetch, planner prompt line, `resources.local` map); pinned `conditions` and `primary_metric` in the brief; pack `plan_checks` and `resource_sources`; `ml_dl` `held_out_issues`; the generic "test cannot reach alpha with these seeds" check | Mac / CI only |
| 3 | `rtool validity submit` and `score`; the card (JSON + Markdown); the eight cases with their `generate.py` and `reference.py` | Mac / CI; the reference scripts run locally |
| 4 | The baseline: deploy a clean commit to Barkla, submit, score, commit the cards | **Barkla, with your OK** |

PR 4's preconditions:
- The Qwen3.6-35B-A3B container and weights must be on Barkla. Memory from 2026-11-20 says both are still missing there.
- A clean deploy from a commit.
- Your approval for the LLM episode GPU time.

## Open questions

1. **Thresholds** for "Stage 0 is done". Proposal: none for the baseline. The baseline is the number
   we compare against, and thresholds are set for Stage 1 once we know where today's tool stands.
2. **Repeats.** Start with 1 per case to learn the cost, then 3?
3. **Case ownership.** Who signs off a case's expected outcome? For Stage 0 the truths are analytic or
   confirmed by a reference script, so you could. From Stage 1, a domain expert.

## Not in Stage 0

- Stage-level cases (replaying one stage on frozen inputs with an assertion). Cheaper per check, but
  full runs come first. `rtool replay` already gives most of it by hand.
- Feeding pack check issues back into the planner's retries.
- Literature-source and reviewer-rubric hooks (designed above, built in Stage 1).
- Any GPU experiment.
- Thresholds, or claims above L3.
