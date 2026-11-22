# Validity card: sysbio

2026-11-22T05:32:22+00:00. Code `1abc7339e8ab`, no git commit; model `Qwen/Qwen3.6-35B-A3B`.

## Headline (confirmed cases)

Runs that reached their case's top level (L3; L1 for open cases):

- **null**: 0 / 3 (0 %, 95 % interval 0–56 %)
- **planted**: 0 / 1 (0 %, 95 % interval 0–79 %)
- **replication**: 0 / 1 (0 %, 95 % interval 0–79 %)

**False claims** on null cases (the report says supported): 0 / 3 (0 %, 95 % interval 0–56 %)

## Highest level reached

| kind | none | L0 | L1 | L2 | L3 |
|---|---|---|---|---|---|
| null | 0 | 3 | 0 | 0 | 0 |
| planted | 1 | 0 | 0 | 0 | 0 |
| replication | 0 | 0 | 1 | 0 | 0 |

## By variant

- ssa-method-null: r0_1_5 0/1; r0_2 0/1; r0_3 0/1

## Failures

- `incomplete_runs`: 3 (ssa-method-null-r0_1_5-r1-1abc733, ssa-method-null-r0_2-r1-1abc733, ssa-method-null-r0_3-r1-1abc733)
- `plan_issue`: 1 (sir-final-size-main-r1-1abc733)
- `stage_failed:fetch`: 1 (vaccination-shrinks-epidemic-main-r1-1abc733)

## Cost

- completion_tokens: 327820
- gpu_episode_hours: 0.776
- llm_calls: 178
- prompt_tokens: 410167
- wall_hours: 13.81

## What this card does not measure

- Cases start at the plan stage with a fixed hypothesis, conditions and primary metric: this measures whether rtool carries out a clear question correctly, not literature search, hypothesis quality or experiment design from an open question (open cases are scored for finishing and consistent code only).
- Every number is for the model and code named under provenance; another model or code means new runs.
- L4 (a new finding is trustworthy) is not measured: that needs replication by people.
