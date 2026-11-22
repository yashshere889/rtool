# Validity card: ml_dl

2026-11-22T05:32:20+00:00. Code `1abc7339e8ab`, no git commit; model `Qwen/Qwen3.6-35B-A3B`.

## Headline (confirmed cases)

Runs that reached their case's top level (L3; L1 for open cases):

- **null**: 1 / 2 (50 %, 95 % interval 9–91 %)
- **open**: 0 / 1 (0 %, 95 % interval 0–79 %)
- **planted**: 2 / 2 (100 %, 95 % interval 34–100 %)

**False claims** on null cases (the report says supported): 0 / 2 (0 %, 95 % interval 0–66 %)

## Highest level reached

| kind | none | L0 | L1 | L2 | L3 |
|---|---|---|---|---|---|
| null | 0 | 1 | 0 | 0 | 1 |
| open | 0 | 1 | 0 | 0 | 0 |
| planted | 0 | 0 | 0 | 0 | 3 |
| replication | 0 | 0 | 0 | 0 | 1 |

## By variant

- random-halves-null: halves_1000 1/1; halves_5000 0/1
- xor-needs-hidden-layer: clear 1/1; noisy 1/1

## Not in the headline: expected outcome not confirmed yet

- label-noise-hurts: 1 / 1 (100 %, 95 % interval 21–100 %)
- softmax-mnist-accuracy: 1 / 1 (100 %, 95 % interval 21–100 %)

## Failures

- `fake_experiment`: 1 (open-topic-main-r1-1abc733)
- `incomplete_runs`: 1 (random-halves-null-halves_5000-r1-1abc733)

## Cost

- completion_tokens: 610749
- gpu_episode_hours: 1.401
- llm_calls: 750
- prompt_tokens: 1.1674e+06
- wall_hours: 13.692

## What this card does not measure

- Cases start at the plan stage with a fixed hypothesis, conditions and primary metric: this measures whether rtool carries out a clear question correctly, not literature search, hypothesis quality or experiment design from an open question (open cases are scored for finishing and consistent code only).
- Every number is for the model and code named under provenance; another model or code means new runs.
- L4 (a new finding is trustworthy) is not measured: that needs replication by people.
