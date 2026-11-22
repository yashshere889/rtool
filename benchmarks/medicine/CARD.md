# Validity card: medicine

2026-11-22T05:32:23+00:00. Code `1abc7339e8ab`, no git commit; model `Qwen/Qwen3.6-35B-A3B`.

## Headline (confirmed cases)

Runs that reached their case's top level (L3; L1 for open cases):

- **synthesis**: 1 / 1 (100 %, 95 % interval 21–100 %)

**False claims** on null cases (the report says supported): 0 / 0 (0 %, 95 % interval 0–100 %)

## Highest level reached

| kind | none | L0 | L1 | L2 | L3 |
|---|---|---|---|---|---|
| synthesis | 0 | 0 | 0 | 0 | 1 |

## Cost

- completion_tokens: 342841
- gpu_episode_hours: 0.407
- llm_calls: 1427
- prompt_tokens: 1.88654e+06
- wall_hours: 0.855

## What this card does not measure

- Cases start at the plan stage with a fixed hypothesis, conditions and primary metric: this measures whether rtool carries out a clear question correctly, not literature search, hypothesis quality or experiment design from an open question (open cases are scored for finishing and consistent code only).
- Every number is for the model and code named under provenance; another model or code means new runs.
- L4 (a new finding is trustworthy) is not measured: that needs replication by people.
