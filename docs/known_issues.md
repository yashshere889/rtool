# Known issues

Open bugs and gaps found on real Barkla runs, with what was seen, the cause where known, and a
proposed fix. Remove an entry once its fix is merged.

Most of these come from run `multi-agent-research-pipeline/20261014-214434-bdf6` (topic: LLM
multi-agent systems that automate research; hypothesis: cross-stage consistency checks added to an
ARIS-style audit cut missed errors by at least 30%). Its third report draft (2026-10-29, waiting
at the `final_report` gate) correctly calls the hypothesis refuted: a −7.5% relative cut against
a required 30%, and false alarms up 4.1 points, within the 5-point limit. Issues 1–5 are still
in that draft.

## Report (Writer and Reviewer)

### 1. Reviewer edits for minor issues make things up — major; fixed on this branch

**Seen.** The Reviewer rewrites a section for every issue aimed at the report, minor ones
included. To answer "doesn't specify what changes were made", it added "documented in version
control (e.g., commit hash [example]), were limited to optimizing token usage and refining
prompt clarity". To answer "does not explicitly link to specific findings", it added "[@yang2026aris]
highlights that the system's reliance on sequential checks introduces vulnerabilities in
cross-stage validation". Neither is in the material.

In the previous round it pasted its own issue text into the sections ("The discussion mentions
limitations but does not quantify…"), and it invented a "quantitative analysis of model size
effects" and a prior-work rationale for the 30% threshold.

**Cause.** Minor points often ask for things the material does not hold (more detail, more
analysis). The prompt rule against this (`reviewer/prompts/edit_section.j2`, added in d5aa1fd)
is not enough for Nemotron. The number check passes, because none of the made-up text contains
a number. The `pasted()` guard catches only verbatim copies.

**Proposed fix.** Edit sections only for `major` issues; leave minor ones open for the
researcher. This needs a change to `Reviewing.review()` (`to_edit`) and to
`test_review_accepts_edits_and_waits_for_the_human`, which expects a minor issue to be edited.

**Status.** Done (a1831e9): only `major` report issues are rewritten; minor ones stay open and show as
"open" in review.md. The demo run and the pipeline test now raise a major issue to show an edit. Still
open: majors can ask for missing material too, and nothing checks that a rewrite's new sentences are backed
by the material (a grounding judge, or rejecting citations attached to new sentences).

### 2. The relative change and the percentage-point change get swapped — minor; fixed on this branch

**Seen.** "a relative reduction of -5.3 percentage points (i.e., -7.5%)". It is the other way
round: −7.5% relative, −5.3 points absolute. Also "−5.3 percentage points lower" (a double
negative).

**Cause.** `Drafting.facts()` lists the two values as `relative change {{rel:…}}, i.e.
{{pts:…}}`. "i.e." suggests they are the same quantity.

**Proposed fix.** Label them "relative change (share of the baseline mean)" and "absolute
change", and phrase a fall as a fall rather than as "a negative reduction".

**Status.** Done (82bc304): `rel:` and `pts:` render without a sign ("7.5%", "5.3 percentage
points"); the facts say "`miss_rate` fell; … relative fall (share of the baseline mean) …, absolute fall …";
the results instructions and the verdict prompt say they are different quantities.

### 3. The Writer misreads the disclosure it is given — minor

**Seen.** "reruns after parser fixes confirmed identical results, ruling out procedural
errors". The researcher's comment said the model's *replies* were identical and only how they
were read changed. The results did change (miss rate 0.720 → 0.704 for the baseline, 0.683 →
0.651 for the enhanced audit), although the conclusion did not.

**Cause.** The model paraphrases free-text feedback loosely. Nothing checks feedback-derived
claims.

**Proposed fix.** Not clear yet. One option: when the code bundle shows hand-edited code or the
results were re-collected, rtool writes the disclosure itself (a fact section, like
reproducibility) instead of leaving it to prose.

### 4. Bare placeholders read as numbers without a noun — minor; fixed on this branch

**Seen.** "The experiment included 16 runs (8 seeds per condition) across 2." Here
`{{conditions}}` was written without "conditions" after it. Elsewhere the abstract listed
standard deviations ("as well as 0.0189 and 0.0175") with nothing to say what they belong to.

**Proposed fix.** In the value meanings, say how to use each placeholder (for example "N
conditions"). Leave `std:` placeholders out of the abstract's and results' lists, since the
tables already show them.

**Status.** Done (b3ca112): `{{conditions}}` and `{{seeds}}` render with their noun ("2 conditions");
a noun typed after them is not doubled; `std:` is left out of the abstract's and results' placeholders.

### 5. The report credits the wrong Slurm job — minor; fixed for new runs

**Seen.** Reproducibility says "Slurm jobs the results came from: 11036397". That job failed
in all 16 tasks; the results came from array 11055870, which was resubmitted by hand.

**Cause.** Run files did not record their job. Fixed in 4fafa72: `harness/run.py` writes
`SLURM_ARRAY_JOB_ID`, and `collect()` prefers it. This run's files predate the fix, so its
report has to be corrected by hand at the gate.

### 6. Each report round sees only the newest comment — minor; fixed on this branch

**Seen.** Round 3 would have lost round 2's requests (the disclosures) had the comment not
repeated them.

**Cause.** `Drafting.draft()` passes `ctx.feedback` (feedback not yet consumed) only. The
Literature agent was changed to use `ctx.earlier_feedback` too (531b14d); the Writer was not.

**Proposed fix.** Give the Writer earlier comments as well, marked as earlier, the way
`literature/agent.py` does.

**Status.** Done (681e68b): `section.j2` lists earlier comments under "Earlier requests, from previous
rounds".

### 7. Researcher feedback goes to every section — minor; partly fixed

**Seen.** In round 2 the code and parser disclosures turned up in the introduction and related
work ("which our study addresses through revised parsing methods").

**Status.** `section.j2` now says to address only the requests that belong in the section
(d5aa1fd), and round 3 was clean. Watch for it on the next run.

## Fixed this week (for reference)

- The verdict came from significance, not from the success criterion: a significant 7.5% cut was
  called support for a 30% target, by both the Writer and the Reviewer. Fixed: one judge call
  applies the criterion, and `rel:` and `pts:` placeholders give the size of each change
  (4fafa72).
- The verdict's checks came back as `[":{"]` after a reasoning-then-format call. Fixed: one
  greedy constrained call, and checks must be sentences (d5aa1fd).
- "p = p = 0.008" and "percentage points percentage points" (4fafa72, d5aa1fd).
- Code the researcher wrote was not disclosed (4fafa72).

## Coder and experiments

### 8. rtool's experiment script cannot run vLLM inside the Nemotron container — major; fixed on this branch

**Seen.** All 16 tasks of job 11036397 failed. HF's modules cache `/opt/huggingface_cache/modules`
is read-only in the image. Then vLLM failed with "Cannot re-initialize CUDA in forked
subprocess".

**Workaround.** The hand-written `experiment.py` sets these before it imports vLLM:

- `HF_HOME`;
- `HF_MODULES_CACHE` (to the run's workdir);
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`;
- `VLLM_ENABLE_V1_MULTIPROCESSING=0`.

**Proposed fix.** The experiment job script sets them (through `APPTAINERENV_`), or the
harness sets them before it imports `experiment.py`.

**Status.** The read-only modules cache is fixed by f74ba19 on initial-build (`writable_caches`), not yet
merged into this branch. `HF_HUB_OFFLINE` was already set. The likely cause of the fork error was rtool's
own `set_seed()`: it called `torch.cuda.is_available()` before the experiment was imported, which
initialises CUDA in the parent. Done (1f84028): `set_seed()` no longer touches `torch.cuda`
(`torch.manual_seed` seeds the GPUs anyway), and GPU runs get `VLLM_ENABLE_V1_MULTIPROCESSING=0` and
`TRANSFORMERS_OFFLINE=1`. Not yet confirmed in the real image on Barkla.

### 9. GPU-only experiments have no smoke path — major

Smoke tests run on CPU (`CUDA_VISIBLE_DEVICES=""`), so an experiment that calls the LLM cannot
be smoke-tested. The Coder answered by writing mock code (random flags) that passed the smoke
test. A deterministic stand-in for smoke mode only (here `RuleAuditor`) works, but nothing
asks for one.

### 10. The model's JSON replies are often malformed — major for LLM experiments

In the first full runs, about 22 of 60 fact-extraction replies and 9 of 60 mapping replies per
run had broken outer braces and were thrown away. That starved the cross-stage checks. A
tolerant parser (it decodes the list after `"key": [` with `raw_decode`) brought this down to
about 4 and 1 per run. LLM experiments written by the Coder should get such a parser by default,
and should report parse failures as a metric.

## Literature and hypotheses

- **Hypothesis rubric saturates:** nearly every hypothesis scores 5/5 on every criterion, so
  ranking does not separate them.
- **S2 debris:** reference-list records with no id, authors, year or abstract enter the corpus
  (49 in this run), and two were kept at checklist score 10.0.
- **Slow downloads:** the download phase took 17 minutes for 63 papers.
