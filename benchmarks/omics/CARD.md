# Validity card: omics

2026-11-22T05:32:21+00:00. Code `1abc7339e8ab`, no git commit; model `Qwen/Qwen3.6-35B-A3B`.

## Headline (confirmed cases)

Runs that reached their case's top level (L3; L1 for open cases):

- **null**: 3 / 4 (75 %, 95 % interval 30–95 %)
- **planted**: 1 / 2 (50 %, 95 % interval 9–91 %)

**False claims** on null cases (the report says supported): 0 / 4 (0 %, 95 % interval 0–49 %)

## Highest level reached

| kind | none | L0 | L1 | L2 | L3 |
|---|---|---|---|---|---|
| null | 0 | 1 | 0 | 0 | 3 |
| planted | 1 | 0 | 0 | 0 | 1 |

## By variant

- pseudobulk-null: seed_1 1/1; seed_2 1/1; seed_4 1/1; seed_5 0/1
- pseudobulk-planted: fold_1_5 0/1; fold_2 1/1

## Failures

- `fake_experiment`: 1 (pseudobulk-null-seed_5-r1-1abc733)
- `stage_failed:code`: 1 (pseudobulk-planted-fold_1_5-r1-1abc733)

## Cost

- completion_tokens: 491014
- gpu_episode_hours: 1.212
- llm_calls: 252
- prompt_tokens: 570158
- wall_hours: 13.461

## What this card does not measure

- Cases start at the plan stage with a fixed hypothesis, conditions and primary metric: this measures whether rtool carries out a clear question correctly, not literature search, hypothesis quality or experiment design from an open question (open cases are scored for finishing and consistent code only).
- Every number is for the model and code named under provenance; another model or code means new runs.
- L4 (a new finding is trustworthy) is not measured: that needs replication by people.
