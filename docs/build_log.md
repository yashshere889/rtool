# Build log: the roadmap stages, one at a time

Plan: [roadmap.md](roadmap.md). Stage 0 design: [stage0_design.md](stage0_design.md).
Started 2026-11-21: build every remaining stage, and test each before the next.

## What "tested" means here

No Barkla access (it needs your approval and an interactive login), no live API calls or data
downloads on the Mac, and no commits unless asked. So a stage passes its **local gate** when all of
these hold:

1. The full suite and lint pass, not just the new tests:
   - `PYTHONPATH=src .venv/bin/python -m pytest` (the main checkout's venv), exit 0.
   - `tests/web` with a venv that has FastAPI (the frontend worktree's), exit 0.
   - `ruff check src tests` and `ruff format --check src tests`.
2. A run of the real engine with the demo model uses the stage's new capability end to end, and the
   validity harness scores it wherever a case applies.
3. Reference scripts that need only the standard library run locally, and match their case's expected value.
4. This log records what was built, the command that proves it, what still waits for Barkla, and the
   stage's files, so each stage can be committed on its own.

The roadmap's **exit tests** need real runs on Barkla, and an expert to confirm each field's ground
truth. They are listed per stage under "Waits for Barkla". The local gate does not replace them.

Parsers for outside services are written to their documentation and tested on fixtures in that
format. Each has a live probe marked `@pytest.mark.barkla` (deselected by default), to run on Barkla:
`pytest -m barkla`.

---

## Stage 0 — validity harness

PR 1 committed as `8ceb2ed` (case format, seeded runs, scorer, statistics calibration). PR 2 and PR 3
share lines in the same files, so they are committed together, as one Stage 0 commit.

### PR 2 — local data, the fixed design in the brief, pack checks (local gate passed 2026-11-21)

Built:
- `ResearchBrief.conditions` and `.primary_metric`. The planner sets what they fix (`follow_brief`:
  seeds, the baseline flag, the primary metric and its direction), and `brief_issues` reports what a plan
  (or a hand edit) still breaks. Validity cases now pin their design through these fields; the
  constraint-text shortcut from PR 1 is gone.
- A generic check: an exact rank test that cannot reach p < 0.05 with the planned seeds is a plan issue
  (Wilcoxon below 6 seeds, Mann–Whitney below 4).
- Pack contract: `plan_checks` (`"module:function"`, loaded when the pack is, so a misspelt one fails
  early) and `resource_sources` (an allowlist). The first check is `ml_dl`'s `held_out_issues`. Pack
  checks are recorded in `plan.issues`, not repaired.
- The `local` resource source: `experiments.local_data: {name: {path, description}}`. It is verified on
  disk, listed in the planner's resources prompt, and copied at fetch time to
  `<resources>/local/<name>-<sha12>`, so changed data never overwrites what a run read.
- Validity case data: `generate_data` runs each case's generator (`python -I generate.py <out> <json args>`)
  at submit time, keeps the output under its content hash in `<runs_root>/.validity-data`, and puts it
  in the run's overlay. Cases also gain `confirmed_by:`.
- API types regenerated (`local` in `ResourceSource`), and the bundle rebuilt (only the stamp changed).

Proved by:
- `tests/unit/test_brief_and_pack_checks.py` (brief fields, rank tests, source allowlist, held-out
  check, local check and copy).
- `tests/integration/test_validity.py::test_a_case_run_with_its_own_data_uses_it_as_a_local_resource`:
  a case's generated data reaches a demo-model run as a `local` resource, through fetch and the
  experiment, and the run scores L3.
- Full suite exit 0, web tests exit 0, ruff clean.

Waits for Barkla: a real planner choosing a `local` resource, and a real fetch of one.

Files: `schemas/common.py`, `schemas/plan.py`, `core/config.py`, `agents/planner/{agent,checks}.py`,
`agents/planner/prompts/resources.j2`, `agents/coder/results.py` (`family_of_tests`),
`agents/tasks/fetch_resources.py`, `tools/resources/{checks,fetch}.py`, `packs/__init__.py`,
`packs/ml_dl/{__init__,checks}.py`, `packs/ml_dl/pack.yaml`, `testing/demo_run.py`,
`validity/{cases,runs}.py`, `frontend/src/api/{openapi.json,schema.d.ts}`, `web/static/build-stamp.json`,
`tests/unit/test_brief_and_pack_checks.py`, `tests/unit/test_validity_cases.py`,
`tests/integration/test_validity.py`.

### PR 3 — `rtool validity`, the card, eight cases (local gate passed 2026-11-21)

Built:
- `rtool validity submit <pack>`: creates and starts every missing case × variant × repeat run of this
  code version, refuses uncommitted code (unless `--allow-dirty`), and skips runs that already exist.
- `rtool validity score <pack>`: scores the runs at rest and writes the card (JSON and Markdown) to
  `<runs root>/.validity-cards/`.
- `validity/provenance.py`: a hash of the installed package identifies the code. Barkla has no git
  metadata, so this is what names a version there. The git commit and a dirty flag are added where git
  exists.
- `validity/card.py`: the headline per kind with Wilson intervals (confirmed cases only), false claims on
  nulls, levels reached, rates per variant, unconfirmed cases apart, failure categories, cost from
  `rtool stats`, and the card's limits.
- Variants can now change the brief (`brief:`, merged), so one null case covers several queue loads.
- Eight cases in `benchmarks/` (see its README).
- References run on the Mac:
  - `mm1-time-in-system`: 4.989 over 8 seeds (formula 5).
  - `two-servers-shorter-wait`: 4.989 against 1.191 (formula 1.190), paired t −105.8.
  - `xor-needs-hidden-layer`: linear 0.729 against MLP 0.994 at noise 0.35, and 0.653 against 0.881 at
    noise 0.7. This showed the first band was wrong: a line gets three of four XOR corners right, so the
    difference is about 0.27, not 0.45. The bands were set from the references.

Proved by:
- `tests/integration/test_validity_cli.py`: `submit` creates the run, a second `submit` skips it, the
  demo model drives it to the end, and `score` writes a card with planted 1/1, cost and provenance.
  Uncommitted code and unknown cases are refused.
- `tests/unit/test_validity_card.py`: Wilson values, card counting, the code hash with and without git,
  and every shipped case loading with its data generated.
- Full suite exit 0, web tests exit 0, ruff clean (now also over `benchmarks/`).

Waits for Barkla: the two torch references (`label-noise-hurts`, `softmax-mnist-accuracy`; their cases
stay unconfirmed until then), and every real run of the eight cases.

Files: `cli/main.py` (validity commands), `validity/{card,provenance,cases,runs,score}.py`,
`testing/demo_run.py` (`demo_case_data`), `benchmarks/**`, `docs/barkla_runbook.md` (PR 4 section),
`tests/integration/test_validity_cli.py`, `tests/unit/test_validity_card.py`,
`tests/integration/test_validity.py`.

### PR 4 — the baseline on Barkla (written up, not run)

This needs your OK for the LLM episodes' GPU time, the Qwen container and weights on Barkla, and a
clean deploy. The exact steps are in [barkla_runbook.md](barkla_runbook.md#validity-baseline-stage-0-pr-4).
Nothing was run on Barkla.
