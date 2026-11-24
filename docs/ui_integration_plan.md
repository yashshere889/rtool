# Plan: bring every rtool feature into the web UI

Status: proposal, 2026-11-24 (main e03d6d8). Nothing built yet.
Rules carried over from the approved UI plan ([frontend.md](frontend.md)): the UI never writes engine
state (actions and inbox uploads only), never calls a language model, adds no frontend dependency
(Node 22.0, Vite 6, Vitest 3), and every phase ends with `scripts/build_frontend.sh`, regenerated types
and a committed bundle (a test fails otherwise). A check-in after each phase; commit only when asked.
Backend items continue the old numbering from B19.

## What was checked (the gaps are real, not guessed)

| Finding | Evidence |
|---|---|
| The UI serves the **research** pipeline only. `study` and `synthesis` runs open on generic stage pages. | `PAGES` in `StagePage.tsx` has 6 agents; no page or text for `protocol`, `collect_data`, `analyse_table`, `synthesis`. |
| Local/demo mode **drops the pack**: `POST /runs` calls `create_demo_run(...)` without `pack`/`pipeline`, and `demo_run.py` builds research runs only. | `web/api/runs.py:111-135`, `testing/demo_run.py:580`. No page for the other pipelines can be tested until this is fixed. |
| Gates `protocol_review`, `data_upload`, `manual_resources` have no copy; they get the generic "Approve / Request changes". | `GATES` in `lib/stages.ts` (6 entries). |
| `data_upload` cannot be answered from the UI: `rtool add-data` copies `data.csv` to the stage inbox and approves. | `cli/main.py:1190`; no upload endpoint except the literature one. |
| The plan editor **cannot represent newer plans**: resource sources `local, geo, cellxgene, biomodels, uniprot, alphafold, pdb` are missing from the dropdown; `unit` / `replicates` (units mode) are not shown. | `RESOURCE_SOURCES` in `lib/plan.ts` (5 of 12); no `replicates`/`unit` in `PlanForm.tsx`/`PlanRead.tsx`. |
| The report never states its **validity level** (L0–L3) or the cases behind it. The roadmap requires it. | No level in `schemas/report.py`, the writer, or `report.py`. Backend and UI both missing. |
| Coder events the Code page needs (`phase_changed`, `smoke_result`, `code_fix`) are still emitted in tool-agent mode (same `Coder` class). New `tool_agent` event (turns, tool calls/errors, denials) is shown nowhere. | `agents/coder/agent.py:408,702,744`; `agents/tool_agent.py:183`. |
| `rtool stats`, refusals, packs, `doctor`, pipelines/agents lists, validity cards: not in the UI. | grep of `frontend/src`. |

## Disposition of every command

| Command | Disposition |
|---|---|
| `new`, `run`, `status`, `approve`, `reject`, `select`, `retry`, `note`, `edit`, `reopen`, `stop`, `list`, `show`, `events`, `add-paper` | **Done** (New research, Overview, stage pages, activity feed, paper upload). |
| `ui` | Is the UI. |
| `add-data` | **Build** (Phase 2). |
| `stats` | **Build**, Overview card (Phase 3). |
| `packs`, `pipelines`, `agents`, `doctor` | **Build read-only** in Settings (Phase 4). |
| `validity score` / cards in `benchmarks/` | **Build read-only** (Phase 5). |
| `replay` | **Build**, "Redo this stage in a copy" (Phase 5). |
| `archive` | **Build** (Phase 5), included at the user's request. |
| `llm-bench`, `validity submit`, `episode` | Left out, as recommended (not yet confirmed by the user): `llm-bench` needs a GPU and the UI is a CPU job; `validity submit` starts many unattended runs; `episode` is internal. |

## Phases

### Phase 1: make the other pipelines testable (backend B19–B20)

* B19: `create_demo_run(..., pack=, pipeline=)`; `POST /runs` passes `pack` in local mode too.
* B20: demo scenarios for `study` (stops at protocol, data upload, final report) and `synthesis`
  (stops at literature, final report), mirroring `build_demo_run(stop_at=...)`, with scripted model answers.
* `tests/web`: one API test per pipeline on the demo runs.
* Exit: `rtool ui --mode local --demo`, choosing a study or synthesis field, makes a run of that pipeline.

### Phase 2: the study pipeline (protocol, data, analysis)

* **Protocol page** (agent `protocol`): the protocol as a document (conditions, unit of analysis,
  primary outcome, sample size, allocation, materials, blinding, checklist, data-file format with a
  downloadable CSV template). Editor: a new `@artifact_editor("protocol")` (B21) that recomputes the
  checklist and refuses edits that break the data format later stages read. Reuse `PlanForm` pieces
  (`ListEditor`, `NumberInput`) and the draft autosave.
* **Collect page** (agent `collect_data`): file picker → `PUT …/collect/inbox` (B22; same pattern as
  `literature/inbox`: size cap, name check, resolved inside the folder) → queued `approve` on
  `data_upload`, as `add-data` does. Shows format problems from `read_table` when the stage stops.
* **Analyse page** (agent `analyse_table`): results per metric and the generated analysis script, reusing
  the Code page's Results tab and diff view. Read-only.
* Gate copy for `protocol_review`, `data_upload`.
* Add the three agents to `PAGES` and `STAGES`.

### Phase 3: synthesis, fetch, and run-level extras

* **Synthesis page**: the review by section; each sentence with its claim ids and papers; a "dropped
  sentences and why" panel; checks. Reuse `QuoteBlock` and `linkCitations`, and link a claim to the
  Paper reader. Read-only unless a synthesis editor is wanted (B23, optional).
* **Fetch page** in place of the generic one: per resource status (`ready`, `deferred`, `manual`,
  `failed`), size against quota, licence. At `manual_resources`, a per-resource upload (B22 pattern).
  Gate copy for `manual_resources`.
* **Plan editor catch-up**: all 12 resource sources with their links and identifier hints; show and edit
  `unit` / `replicates`; `PlanCheck` already runs server side. Add a test with a units-mode plan.
* **Overview**: a "Where the time went" card from `stats_for_run_dir` (B24: `GET …/stats`), and the
  `tool_agent` event (turns, errors, denials) in the Code page's Tests tab.
* **Refusals**: when a run or plan is refused, show the reason as a clear card (today only an error toast
  via `lib/events.ts`). Reading the existing events is enough; no new backend.

### Phase 4: operator views in Settings (read-only)

* B25: `GET /packs`, `/pipelines`, `/agents`, `/doctor` wrapping the code behind the CLI commands.
  `doctor` with `network=False` by default; a button runs it with the network check.
* New Research: the field list says which pipeline each pack uses and what it needs (GPU, a person,
  no experiment), using the `packs` data. Show the per-pack GPU-hour cap before the user types a budget.
* Settings gets tabs: Server, Keys, Fields, Pipelines & agents, Checks.

### Phase 5: validity, replay and archive

* B26: the report states its validity level and the cases it was measured on (from the pack's card);
  `ReportView` carries it. This is a roadmap requirement the engine does not do yet.
* Report and Review pages: a level badge ("Measured at L2 on 4 cases") and a link to the card.
* Validity page (read-only): cards from `benchmarks/<pack>/` with pass rate per case kind, false-positive
  rate, failure categories, model, code version.
* `replay`: a "Redo from here in a copy" action on a stage page (B27: `POST …/replay`, which calls
  `replay_run` and kicks the copy). Needs an OK on the cost: it makes a whole new run.

* **Archive** (B28): an "Archive this run" button on the Overview. `POST …/archive` calls `archive_run`
  (the code behind `rtool archive`) and reports the file count, size and the files left out for being
  too large. Whole-run tars can be huge, so:
  * it runs as a background task in the server (a CPU job), never inside the request, and the page polls
    its state; one archive per run at a time;
  * it needs `paths.archive_root` set; if not, the button is disabled and says why (as the CLI does);
  * the single-file cap defaults to 200 MB as in the CLI and is shown before starting, with the run's
    size; the user may lower it, not remove it;
  * it only reads the run and writes the tarball under `archive_root`, never engine state, so it is not an
    action. The tarball is not served for download (the code zip and report bundle already are); the page
    shows its path on the cluster. Open question: whether the existing 8 MB / 64 MB request caps matter
    here (they do not: nothing is uploaded).
  * Test: API test on a demo run with a small `archive_root`; page test for the disabled and running states.

### Phase 6: end-to-end coverage and docs

* e2e today walks the research pipeline only. Add one walk each for `study` (including data upload) and
  `synthesis`; keep the axe checks in light and dark.
* Update `docs/frontend.md` (pages table, "Adding a page" already describes the steps).

## Per-page checklist (from `frontend.md`)

API module in `web/api/` → regenerate types → hooks → page under `src/pages/<name>/` → `PAGES` +
`STAGES` + `GATES` → API test on a demo run → Vitest page test → e2e step if it has a checkpoint.

## Which outputs get an editor

| Output | Editor | Why |
|---|---|---|
| `protocol` | Yes (B21) | It is what the researcher then carries out; they will want to change it. |
| `synthesis` | Optional (B23) | Edits must keep every sentence tied to a verified quote; ask before building. |
| `data.csv`, fetch results, analysis, stats, packs, doctor, validity cards | No (read-only or upload only) | Derived or operator data. |

## Order and size

Phase 1 first (it unblocks testing). Then 2 and 3 are the bulk. Phases 4 and 5 are small. Phase 6 runs
inside each phase as pages land. Rough size: 8 new pages or panels, 9 backend items (B19–B28; B23 optional), no new dependency.

## Not verified

* Real Barkla runs of the study and synthesis pipelines (only demo and local gates exist).
* Whether `edit` can safely re-run the stages after `protocol` (checked only by reading the planner's editor).
