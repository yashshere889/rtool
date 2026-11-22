# Validity card: generic

2026-11-22T05:32:18+00:00. Code `1abc7339e8ab`, no git commit; model `Qwen/Qwen3.6-35B-A3B`.

## Headline (confirmed cases)

Runs that reached their case's top level (L3; L1 for open cases):

- **null**: 0 / 3 (0 %, 95 % interval 0–56 %)
- **planted**: 0 / 1 (0 %, 95 % interval 0–79 %)
- **replication**: 0 / 1 (0 %, 95 % interval 0–79 %)

**False claims** on null cases (the report says supported): 2 / 3 (67 %, 95 % interval 21–94 %)

## Highest level reached

| kind | none | L0 | L1 | L2 | L3 |
|---|---|---|---|---|---|
| null | 0 | 0 | 3 | 0 | 0 |
| planted | 0 | 0 | 1 | 0 | 0 |
| replication | 0 | 1 | 0 | 0 | 0 |

## By variant

- rng-choice-null: load_50 0/1; load_80 0/1; load_95 0/1

## Failures

- `fake_experiment`: 1 (mm1-time-in-system-main-r1-1abc733)
- `plan_issue`: 4 (rng-choice-null-load_50-r1-1abc733, rng-choice-null-load_80-r1-1abc733, rng-choice-null-load_95-r1-1abc733, two-servers-shorter-wait-main-r1-1abc733)

## Cost

- completion_tokens: 381569
- gpu_episode_hours: 0.945
- llm_calls: 215
- prompt_tokens: 492107
- wall_hours: 6.735

## What this card does not measure

- Cases start at the plan stage with a fixed hypothesis, conditions and primary metric: this measures whether rtool carries out a clear question correctly, not literature search, hypothesis quality or experiment design from an open question (open cases are scored for finishing and consistent code only).
- Every number is for the model and code named under provenance; another model or code means new runs.
- L4 (a new finding is trustworthy) is not measured: that needs replication by people.
