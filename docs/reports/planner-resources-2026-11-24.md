# Experiment planner: where it got datasets, models and code wrong, and why

Report date 2026-11-24. Read-only analysis of every `experiment_plan` artifact under `~/scratch/rtool` on Barkla.
Extraction script: [planner_resources_dump.py](planner_resources_dump.py) (run it on Barkla with one output path argument).
Nothing on Barkla was changed.

## 1. Summary

**The planner is good at copying a resource that the brief or the config already names. It is poor at choosing one itself.**

- Where the brief or `experiments.local_data` named the data (MNIST from torchvision, `xor_blobs`, `cells`, `motif_sequences`), the plan listed it correctly, with exact identifiers.
- Where the planner had to choose a dataset or model, it often invented an identifier from memory. It also listed its own code and outputs as "resources", picked models that do not fit the hypothesis, and ignored free-text constraints in the brief.
- Several of these are already fixed on `main`. The ones that are still open are in §6, in rank order.

**The run that most likely prompted this report is `best-language-model-under/20261122-073939-5a90`.**
It shows two problems that are still open:

1. The plan stage was approved with 3 unresolved resource issues (`hfl/ceval`, and two models written as a size description, "8B-parameter BF16 transformer model").
2. The researcher then typed three corrections (use Nemotron-3.5-Bolt-E4B and Raon-1-nano, drop C-Eval Mini). They were attached to the **fetch** stage and never read. `fetch.json` shows all three with `consumed_by_attempt: null`, and the plan stayed at attempt 1. Fetch re-ran the same three failing resources 3 more times (10:07, 10:08, 10:11) and failed each time.

## 2. What was examined, and how to read the numbers

| Item | Count |
|---|---|
| Plan files (`experiment_plan/v*.json`) found | 84 |
| Distinct planner outputs (by producer token; the rest are hand edits and replay copies) | 38 |
| Resources in those 38 plans | 95 |
| Resources the checker could not verify (`verified=false`) | 25 |
| Plans with at least one open issue recorded | 22 of 38 |

The 84 files are not 84 planner decisions. Seven `verify-audit-fixes` runs and ten `bdf6-*` replay runs share the same early plans. The 38 distinct outputs come from these groups:

| Group | Distinct plans | Notes |
|---|---|---|
| Real topics (ac16, bdf6, fin-volatility, verify-vllm-cuda, best-language-model-under) | 11 | Hardest cases: the planner had to choose resources itself |
| `verify-audit-fixes` (label smoothing) | 3 | Same brief, three planner versions |
| Roadmap benchmark (`1abc733`, synthetic briefs) | 24 | Briefs state exactly what to use. The 6 `study` plans come from the study agent, not the planner, so they are not counted here |

Caveats:

- Runs span 2026-10-10 to 2026-11-23, with different planner code and three different planner LLMs (Nemotron Nano 12B, Qwen3-Coder-Next, Qwen3.6-35B-A3B). Each failure in §4 has a status of **fixed since**, **open** or **unknown**, checked against `main` (`git log` for `planner/`, `tools/resources/`).
- "Verified" in a plan means the checker found the identifier. It does not mean the resource fits the task (see F6).
- I did not open the rendered LLM prompts. Statements about what the planner "was told" come from the templates in `src/research_tool/agents/planner/prompts/` and from run files.
- I did not test whether verified Hugging Face datasets load under the installed `datasets` version.

## 3. Where the planner was correct, and why

| Run | What was right | Why it worked |
|---|---|---|
| Roadmap `ml_dl`: label-noise, open-topic, random-halves ×2, softmax-mnist | MNIST via `torchvision`; fetched as `ready` | Brief says "MNIST from torchvision; download nothing else" |
| Roadmap `xor-needs-hidden-layer` clear and noisy | `local: xor_blobs`, exact name, `provided locally (2 files)` | The name came from `local_data` and the prompt says to use it exactly |
| Roadmap `omics` ×6, `protein` | `local: cells`, `local: motif_sequences` | Same |
| Roadmap `sysbio`, `generic` simulations | numpy only; no invented dataset | Brief says "nothing to download" (but see F4: two plans still invented one) |
| Roadmap `study` ×6 | `local: study_data`, shown as unverified | Expected: data arrives at the data checkpoint. The "NOT FOUND" label is a display false alarm |
| `verify-audit-fixes` plan v0003 (brief2) | `torchvision: MNIST`, `torchvision: CIFAR-10`; both fetched `ready` | Brief says "MNIST or CIFAR-10" |
| `best-language-model-under` | `sjyuxyz/openwebtext-subset-1M-rows` (3.0 GB, fetched `ready`); `pip: vllm` deferred | The repair round gave "real ids that may fit" after `skeskinen/openwebtext_10k` returned 404; the planner took the suggestion. Also see F1 |
| `fin-volatility/qwen-garch-1` | `pip: arch`, `pip: statsmodels` | Well-known packages |
| `fin-volatility/qwen-garch-2` v0002 | Ken French factors URL, `ready` | The researcher's feedback named it (see F6) |
| `bdf6` (hand edit v0002) | `ccdv/pubmed-summarization` parquet, `ready` | The researcher replaced the planner's choice (see F1) |

The pattern is the same in every success: the correct identifier was already in the brief, in `local_data`, in a researcher's note, or in the checker's own "real ids that may fit" list. The planner rarely produced a correct, non-obvious identifier on its own.

## 4. Failure modes

Every row is a resource in a planner output (`producer=planner`). Run dates are plan creation dates.

### F1. Invented identifiers (still partly open)

The LLM writes an identifier from memory. The checker then reports it is not found.

| Run (date, planner LLM) | Chosen | Result |
|---|---|---|
| best-language-model (11-23, Qwen3.6) | `hfl/ceval` | 404; **never replaced** (see F7) |
| best-language-model | `skeskinen/openwebtext_10k` | 404; replaced in repair by `sjyuxyz/openwebtext-subset-1M-rows` |
| qwen-garch-2 (11-19, Qwen3-Coder) | `jpmorganchase/spy-5min-2000-2024` (HF dataset) | 404; the repair rounds ended with `manual` |
| qwen-garch-2 | `pip: realized` | 404 on PyPI |
| qwen-garch-2 | Yahoo `query1.finance…/v7/download` | HTTP 401 |
| verify-vllm-cuda (11-15, Nemotron) | `google/gsm8k`, `multi-arith` | 404 (the real id is `openai/gsm8k`; the later plan used the short name `gsm8k`, which resolved) |
| bdf6 (10-21, Nemotron) | `arXiv` (HF dataset) | 404; the researcher replaced it by hand |
| bdf6-plan-qwen (11-16, Qwen3-Coder) | `HuggingFaceM4/scientific_papers`, then `HuggingFaceH4/openai_summarize_tldr` | 404 both times; ended in a plan with 5 of 7 resources unverified |

**Why.** The resource step is one LLM call that asks for identifiers it must "be confident exist". Nothing lets it look them up first. The checker runs only after the list is written, and `repair_rounds` defaults to 1 (`PlannerConfig`, `planner/agent.py:73`), which allows 2 attempts in total. A repair that fixes one wrong id often breaks another: in best-language-model the first repair fixed OpenWebText, the second round then listed `hfl/ceval` and ran out of attempts.

**Status.** The checker's "real ids that may fit" suggestions (`HubChecker.similar`) already work and fixed one case. The round budget is **open**: the config allows up to 3, the default is 1.

### F2. Models: wrong kind, wrong size, or a description instead of an id (open)

| Run | Chosen | Problem |
|---|---|---|
| best-language-model | `local: "8B-parameter BF16 transformer model"`, `local: "4B-parameter BF16 …"` | A model written as a size description with source `local`. The prompt says `local` is only for a dataset the researcher provided; the planner used it anyway. Both failed at fetch |
| verify-vllm-cuda (Nemotron) | `manual: Nemotron Nano 12B`, then `huggingface: meta-llama/nemotron-H3-12B` | Brief gave the exact id `nvidia/NVIDIA-Nemotron-Nano-12B-v2`. That model is in `~/scratch/rtool/hf/hub/models--nvidia--NVIDIA-Nemotron-Nano-12B-v2`. Brief also said "do not list it". The cached-model hint reached the prompt only from `fda738c` (2026-11-20), after this run |
| bdf6 v0001 and plan-nemo | `google-bert/bert-base-uncased` as "language model" for an LLM audit | BERT is an encoder; it cannot produce flags and reasons. The hypothesis's own controls name Nemotron Nano 12B v2 |
| bdf6-plan-qwen | `meta-llama/Llama-3.2-3B-Instruct`, "replacement for unavailable Nemotron" | The checker marked it verified but its note says gated (needs `HF_TOKEN`). It also silently changes the controlled variable of the hypothesis |

**Why.** A model is checked only for existence, never for fit (kind, size, access, whether it is already on the cluster). The model slot in `ResourceChoice` accepts any `source`, and the checker accepts `local` for models. The planner has no "pick the model" step; it gets a list of cached models in the prompt and is trusted to use it.

**Status.** Cached models in the prompt: fixed since 11-20 (I could not confirm that the 11-23 prompt actually named the Nemotron model it should have). The rest: **open**.

### F3. Torchvision names are never checked at plan time (open)

In `HubChecker.check`, any source that is not handled above falls through to `ResourceCheck(True, "a torchvision dataset: downloaded with torchvision when the plan is approved")` (`tools/resources/checks.py`, last line of the `try` block). So the plan page shows "found" for any torchvision name.

Evidence: the `verify-audit-fixes` plan v0001 (11-12) listed `CIFAR10` and `ImageNet100`, both `verified=True`. torchvision has no `ImageNet100` class (it has `ImageNet`, which cannot be downloaded automatically). Since `a776017` (11-16) fetch resolves the name inside the experiment container and fails with `torchvision has no dataset 'ImageNet100'`, but only after the plan has been approved.

The brief for that run said "One small CNN or MLP on MNIST or CIFAR-10". `ImageNet-100` first appeared in the **selected hypothesis**'s `datasets` field, so the planner inherited it (see F5).

**Status.** Plan-time check: **open**. Fetch-time check: fixed since 11-16.

### F4. The planner lists its own code and outputs as resources (mostly fixed)

| Run | Listed as a resource | Effect |
|---|---|---|
| ac16 (10-12, Nemotron) | `manual`: "CS:GO multi-agent environment", "MADDPG implementation", "Training/evaluation script" | The three folders under `resources/manual/` are still **empty**. The Coder's `experiment.py` loads them by name but never uses them, and computes metrics from formulas (`# mock logic for demonstration`; `task_success_rate = 1.0 - (n_agents - 2) * 0.1`). The run produced results and a report from invented numbers |
| bdf6 v0001, plan-nemo, plan-qwen | `manual` custom pipelines (3 to 5 per plan) | Flagged as issues; some had to be rewritten by hand |
| roadmap mm1 | `manual: queue_simulation_results.csv`, "Simulation Output Data" | An empty output file as a dataset; scored `fake_experiment` |
| roadmap vaccination | `local: results.json` | The run's own output as an input; **fetch failed** |
| verify-audit-fixes plan-limits | `model manual: small_cnn` | The model is trained from scratch; there is nothing to provide |
| verify-vllm-cuda | `manual: "Prompt generation and evaluation script"` | Own code |

**Why.** Two rules push the model this way: the old plan check required a dataset for every run ("no dataset is listed", 6 roadmap simulations), and before 11-21 nothing told it to leave out its own code. Faced with "a dataset is required" for a simulation, the model invented one.

**Status.** The prompt rule "do not list the experiment's own code" and the manual-code issue are on `main` now (I did not trace which commit added them). "A dataset only for data analyses": `f0adbd2` (11-22). Fetch stops at a `manual_resources` gate if a manual folder is empty. **Fixed on main but not re-run on Barkla** (the roadmap runs here predate `f0adbd2`). Still open: `manual` is allowed for datasets and models, and the gate text offers "approve without them, and the Coder will try to build them in code", which is the same road that led to ac16.

### F5. Datasets enter in the hypothesis, and the planner inherits them (open)

| Run | Selected hypothesis `datasets` | What the planner did |
|---|---|---|
| ac16 | "CS:GO simulation environment from arxiv:2509.06355" | `manual` CS:GO env (F4) |
| best-language-model | "C-Eval mini subset", "OpenWebText sampling…", "Structured instruction prompts…" | `hfl/ceval`, OpenWebText subset |
| qwen-garch-1/2 | "S&P 500 daily returns (publicly available via Yahoo Finance or similar)"; "5-minute intraday data" | Yahoo URL; a non-existent HF dataset |
| verify-vllm-cuda | "GSM8K", "MultiArith" | Both listed, although the brief said **"No dataset: generate 40 … problems in code"** and "do not list [the model] or vLLM" |
| verify-audit-fixes | "CIFAR-10", "ImageNet-100" | Both listed, although the brief said MNIST or CIFAR-10 |

**Why.** The hypothesis stage writes dataset names as free text, with no check that they exist or are allowed. The planner gets the hypothesis, the brief's constraints (as prompt text) and the checker. `brief_issues` enforces only the structured `brief.datasets` list (`planner/checks.py:364`). Free-text constraints such as "no dataset" or "MNIST or CIFAR-10" are in the prompt but are not checked.

**Status.** **Open.** A hypothesis can still carry a dataset that violates the brief, and nothing downstream blocks it.

### F6. "Verified" is not "fit for the task" (open)

| Example | What the check said | What was actually wrong |
|---|---|---|
| qwen-garch-1: `finance.yahoo.com/quote/%5EGSPC/history` as the S&P 500 dataset | reachable, `verified=True` | It is an HTML page. The Coder spent 4 repair rounds on `ParserError: Expected 1 fields in line 17` and the stage failed. Fixed since: HTML-page check (in the experiment branch by 11-19, in `main` by `e94fdb0`, 11-22) |
| qwen-garch-2 v0002: Ken French "Research Data Factors (daily)" for "realised variance" | reachable, `ready` | The file holds daily factor returns (Mkt-RF). The hypothesis was about realised variance from intraday data; the substitute is feasible but weaker. This was the researcher's choice after three failed attempts, not a silent planner choice |
| bdf6 v0001: `PubMed` | verified, **0.0 GB** | Size 0.0 GB should have raised a flag. Whether it loads with the current `datasets` library was not tested |
| bdf6 plan-nemo: `scientific_papers` | verified, 26.9 GB | Legacy dataset; loadability not tested here |
| bdf6-plan-qwen: gated Llama | verified | Needs a token and accepted terms; the note says so but the issue list does not |
| plan-qwen: three pipelines + answer-key generator as `manual` | counted as 4 issues | Same as F4 |

**Why.** `ResourceCheck(exists, license, size_gb, note)` answers "does it exist and how big". Nothing checks kind, size sanity, access, format or relevance to the hypothesis.

### F7. Unresolved issues do not stop the gate, and corrections to a failed fetch go nowhere (open)

- **Plan gate.** In best-language-model, `plan.json` shows `gate.resolution: approved`, `by sgyshere (web)`, at 09:14:49, 3 minutes after the gate opened (09:11:25) with 3 issues still open. The plan markdown puts "Unresolved issues (please check)" first, but nothing in the engine or web API code I searched blocks approval on `plan.issues`. In `bdf6-plan-qwen` v0003 the plan reached the gate with 5 open issues (not approved there).
- **Fetch stage.** `fetch_resources.py` never reads `ctx.feedback`. Its failure message tells the researcher to correct the plan only when the failed resource is a torchvision one (`renamed = …`). For Hugging Face or `local` failures the message says "Fix it (e.g. HF_TOKEN for gated data)" and "resume the stage". A note attached to the fetch stage (UI "Retry" with a note) is stored but never consumed. In this run, three notes were stored and ignored; the plan was never re-run.
- **Result.** The researcher's own correction (named models, no C-Eval) was lost 3 times, and the run stayed at `fetch failed`.

**Status.** **Open.** This is the single most visible cause of "the agent didn't pick the correct resources" in the run you used.

### F8. Inconsistency on identical briefs (observation)

The two `xor-needs-hidden-layer` plans (clear and noisy) share one brief. One lists `torch`, the other `tensorflow`; `tensorflow` is not in that run's `experiments.packages` (the list contains `torch`, not `tensorflow`). Both runs finished every stage, so this did not fail a run, but it shows the package choice is not constrained to what the experiment environment has. The `verify-audit-fixes` v0001 plans look identical across 7 runs only because they are copies of one planner output.

### F9. Planner LLM comparison, one run each (weak evidence)

On the same hypothesis (bdf6, re-planned 11-16): Nemotron Nano 12B produced 3 resources, 1 unverified. Qwen3-Coder-Next produced 7 resources, 5 unverified, 3 of them marked `manual` code. With n=1 per model, this is a lead, not a result.

## 5. Failure counts by cause (38 distinct planner outputs)

| Cause | Plans affected | Status |
|---|---|---|
| Invented / wrong identifier (F1) | 8 | partly open (repair budget) |
| Model wrong kind / description / ignored brief (F2) | 5 | open |
| Torchvision name not checked (F3) | 1 (3 variants) | open at plan time |
| Own code or output listed as resource (F4) | 12 | mostly fixed, not re-run |
| Dataset inherited from hypothesis against the brief (F5) | 6 | open |
| "Verified" but unfit (F6) | 5 | open |
| Gate accepts issues; fetch ignores corrections (F7) | 2 runs | open |

Counts are approximate where one plan has several causes.

## 6. Open problems, ranked

1. **Corrections on a failed fetch must reach the plan (F7).** Smallest fix: when fetch fails on a non-torchvision resource, print the plan-edit instruction (as it already does for torchvision), or send a fetch-stage note back to the plan stage as feedback. This is the problem the researcher met.
2. **Plan approval with unresolved resource issues (F7).** Require an explicit acknowledgement, or block approval while any resource is `NOT FOUND`, `manual` model/dataset, or gated.
3. **Check hypothesis datasets against the brief before the plan (F5).** Free-text constraints ("no dataset", "MNIST or CIFAR-10", "do not list the model") are only in the prompt. Either parse them into structured fields or add a check of the plan's resources against the brief text.
4. **Look up before choosing, and allow the repair loop to converge (F1).** Search the Hub for each requested name first and show the candidates in the resources prompt, instead of validating guesses afterwards. Raise `repair_rounds` from 1; the config allows 3.
5. **Models need their own resolution step (F2).** Reject `source: local` and generic descriptions for models. Flag gated models and models whose family does not fit (encoder vs generator). Prefer the cluster's cached models when the brief names one.
6. **Check torchvision names at plan time (F3).** Share the name-matching logic from `TORCHVISION_SCRIPT` with the plan check, or keep a static list of downloadable torchvision dataset names.
7. **Re-run the Stage 0–6 benchmark to see whether `f0adbd2` fixed F4** (no dataset for simulations, no copied ablations). The cards on Barkla are from before the fix.
8. **Sanity-check "verified" (F6).** Flag a 0.0 GB size, and a gated or script-based dataset, at plan time.
9. **Build a small resource-selection benchmark** from this data: for each real topic above, the right dataset and model are known. Score identifier existence, fit, and brief compliance. The dump script produces the starting data.

## 7. Appendix: runs and how each ended

| Run | Date | Planner LLM | Final plan resources | Fetch | Outcome |
|---|---|---|---|---|---|
| ac16 (multi-agent pipeline) | 10-12 | Nemotron | 3 `manual` | `manual`, folders empty | Code used mock formulas; report from invented numbers |
| bdf6 (LLM audit) | 10-21 | Nemotron | planner v0001: HF `arXiv` (404), `PubMed`, BERT, manual script; researcher replaced with `ccdv/pubmed-summarization` | `ready` | Ran to review; human fix worked |
| verify-20261112 ×7 (label smoothing) | 11-12 | Nemotron | `CIFAR10`, `ImageNet100` (torchvision, "verified") | `deferred` | Plan stage then fixed to MNIST/CIFAR-10 in v0003; one run failed at smoke test |
| verify-vllm-cuda | 11-15 | Nemotron | GSM8K, MultiArith, manual Nemotron, manual script; later `gsm8k` + `meta-llama/nemotron-H3-12B` | stale | Researcher edited out the datasets; reviewer later found identical metrics across conditions |
| qwen-garch-1 | 11-17 | Qwen3-Coder | Yahoo HTML page, `arch`, `statsmodels` | `ready` (HTML) | Code stage failed with `ParserError` |
| qwen-garch-2 | 11-19 | Qwen3-Coder | 3 failed planner attempts, then Ken French URL | `ready` | Plan back at the gate with 2 reviewer issues |
| bdf6-plan-qwen / plan-nemo | 11-16 | Qwen3-Coder / Nemotron | see F1, F2, F9 | stale | Experiments in planner comparison |
| roadmap `1abc733` ×30 | 11-21/22 | Qwen3.6 | 27 plans right; mm1 and vaccination invented a resource; xor-noisy listed `tensorflow`; 6 `study` unverified by design | `done` except vaccination `failed` | Card categories: 3 `fake_experiment`, 6 `plan_issue`, 5 `incomplete_runs`, 2 stage failures |
| best-language-model-under | 11-23 | Qwen3.6 | OpenWebText subset (ok), `hfl/ceval` (404), two generic `local` models, `vllm` | `failed` ×4 | Stuck; researcher's corrections ignored (F7) |
