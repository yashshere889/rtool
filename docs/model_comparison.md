# Comparing LLMs in rtool (Nemotron Nano 12B v2 vs Qwen3-Coder-Next)

The question: for the same inputs, which model gives better stage outputs? Stages are compared one at a
time on **frozen inputs**: `rtool replay` copies a finished run and redoes one stage in the copy, so the
only difference between two replays is the model. Prompts, profiles (temperatures, token limits),
schemas and thresholds are identical; `configs/barkla-qwen3-coder-next.yaml` changes only the server and
the reasoning flag.

Since 2026-11-20 the default model is Qwen3.6-35B-A3B; Nemotron is the overlay
`configs/barkla-nemotron.yaml`. Both models are open and run on Barkla; nothing here calls a hosted LLM, including as a judge.

## Differences to keep in mind when reading results

* Qwen3-Coder-Next has no thinking mode, so the `reason`/`code`/`judge` profiles run without a chain of
  thought for it (`reasoning_control: none`). Nemotron thinks in those profiles. That is part of what is
  being compared, not a bug. Structured calls keep the same draft-then-format shape for both.
* FP8 weights, not BF16 (the model card's numbers are for BF16).
* One replay per model per stage is one sample. Repeat (`--tag qwen2`, `--tag nemo2`) before believing a
  small difference; Nemotron at temperature 0.6 and Qwen at 0.0 (the profiles' values) both vary.
* The profiles now hold Qwen3.6's card values (temperature 1.0 / 0.95, 0.7 / 0.8 non-thinking), so both models
  replay with those; the old Nemotron values (0.6 / 0.95, greedy) are no longer in effect.
* Sampling follows the profiles, not the Qwen3-Coder-Next card's recommended temperature 1.0 / top-p 0.95 / top-k 40.
  A second round with those values is a separate experiment.

## One-time setup on Barkla (needs your interactive login)

```bash
# 1. A container with vLLM >= 0.15 (the Nemotron image has 0.11.0). Run on a viz node.
bash ~/"Research tool"/scripts/build_vllm_sif.sh v0.15.0

# 2. The weights (~80 GB) into the shared Hugging Face cache, as a 1-core job from scratch.
cd ~/scratch/rtool/logs && RTOOL_MODEL=Qwen/Qwen3-Coder-Next-FP8 sbatch ~/"Research tool"/scripts/fetch_model.sh
```

If the image tag does not exist, take the newest `vllm/vllm-openai` tag >= v0.15.0 and change
`llm.server.container` in the overlay to match.

## Running a stage with both models

Pick a source run that is at rest and has the stage's inputs (for example
`multi-agent-research-pipeline/20261014-214434-bdf6`). Replay each stage under both models:

```bash
OVERLAY=~/"Research tool"/configs/barkla-qwen3-coder-next.yaml
NEMO=~/"Research tool"/configs/barkla-nemotron.yaml
R=multi-agent-research-pipeline/20261014-214434-bdf6
for stage in hypotheses plan code report; do
  rtool replay $R --stage $stage --tag nemo --config "$NEMO"    # Nemotron
  rtool replay $R --stage $stage --tag qwen --config "$OVERLAY" # Qwen3-Coder-Next
  rtool run $R-$stage-nemo
  rtool run $R-$stage-qwen
done
```

Each copy stops where the stage's own gate opens (hypotheses and plan at their checkpoints, code at
`gpu_runs` before any GPU time, report/review at `final_report`), so no experiment is run. Start with
`hypotheses` on Qwen as the smoke test: if vLLM 0.15 rejects a flag (the overlay keeps
`--safetensors-load-strategy` and `--structured-outputs-config` from barkla.yaml), the episode
records the server error and `rtool stats <run>` names it.

Replay copies the whole run directory (papers, texts, datasets), so use runs whose directory is a few
GB, and delete the copies afterwards.

`literature` can be replayed too, but it searches and downloads again, so its inputs are not frozen.

## What to compare

Objective, per run, from `rtool stats <run>`: LLM calls, tokens, latency, failed or retried structured
calls, server start time, queue wait. Then per stage, read both outputs side by side (`rtool show <run>
<artifact>`) with the model names hidden, and score them by hand:

| stage | artifact | check |
|---|---|---|
| hypotheses | `hypothesis_set` | falsifiable, specific variables and metric, grounded in cited papers, distinct from each other |
| plan | `experiment_plan` | passes the planner's own checks; resources exist; compute fits the partition; analysis matches the hypothesis |
| code | `code_bundle`, `experiment_results` | smoke test passes, number of fix attempts, mock-experiment warnings, seeds actually used |
| report | `report`, `review` | numbers match results, verdict follows the success criterion, no invented citations |

Judge by hand or with the existing checks; do not use either model to grade the other, because each
would favour its own style.
