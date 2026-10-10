# The web UI

`rtool ui` serves a browser interface for researchers: start a study from a question, follow
every stage live, read what the agents found, edit what they produced, and make the decision at
each checkpoint. Everything the UI shows comes from the run directory; everything it changes
goes through the run's action inbox, exactly like the CLI. The UI never calls a language model.

## Pages

One page per stage, plus the run overview. A stage whose agent has no page of its own gets a
generic page (its checkpoint, problems, every version of its outputs as JSON, activity, notes).

| Page | What you see | What you can do |
|---|---|---|
| Projects | every run with its stage dots and a one-line status | open one, start new research |
| New research | question, review mode, optional questions, constraints, start queries, field, GPU-hour cap | start |
| Overview | open checkpoints, failures, stage stepper, activity feed, queued decisions, usage | decide, retry, undo, resume, stop |
| Literature | a card per paper (relevance, how it was found, read in full or not, AI summary), filters; Map, Gaps, State of the art, full-text Search | open the paper, Summary / Methodology / Key findings / Limitations / Gaps / Datasets / References / Cite, upload a PDF you have, mark papers Keep or Drop, approve or request changes, reopen the checkpoint later |
| Paper reader | the PDF or web copy, or the text with every verified finding (or a quoted passage) marked | jump from a finding to its sentence |
| Hypotheses | the shortlist side by side: design, novelty, feasibility, scores and judges' comments, papers used, citations with "show in paper", references; a comparison table; the other drafts and why they were set aside | choose, edit and choose (with a note for the planner), ask for new ones, change your choice later |
| Experiment plan | the plan as a document; datasets and models with links; compute against the budget; downloads after approval | edit any part (form or JSON) with live checks, see what applying redoes, approve |
| Code & runs | progress strip; every file; smoke tests; what the tool repaired; a condition × seed grid of runs with logs; results per metric | edit `experiment.py` (tested again before any GPU time), compare with the generated code, approve GPU time, cancel jobs, download the code |
| Report | the report with linked citations, figures and references | edit prose sections (placeholders and citations as chips, insert pickers), print or save as PDF, download Markdown, BibTeX or everything |
| Review | verdict, scores against the bar, automated checks, each issue and what became of it, revision rounds, the final report, the reviewer's edits as a diff | approve and finish, request another round, edit the final report |

What the agents wrote is labelled. Findings carry their quote and whether it was found in the
paper; AI-written digests and summaries say so and flag numbers the paper does not contain;
report sections say whether rtool, the writer, the reviewer or you wrote them.

## How a click reaches the engine

```
browser ──POST /api/runs/<p>/<r>/actions──▶ checks (now, against the run's state)
                                            │ writes actions/<time>-<id>.json (the inbox)
                                            ▼
                                     kick: an episode job (Slurm) — or, locally, a thread
                                            │ applies queued actions under the run lease
                                            ▼
                                     new artifacts / stage states / events.jsonl ──SSE──▶ browser
```

* Decisions: `approve`, `reject` (with comments; on the literature checkpoint also `include` /
  `exclude` lists of paper ids), `select` (with an optional `edit` of the chosen hypothesis),
  `retry`, `note`, `reopen` (a finished stage's checkpoint, e.g. to change the chosen hypothesis
  or add papers).
* Edits: `edit` replaces an output with your version (`experiment_plan`, `report`,
  `final_report`). The owning agent's editor (`core/editing.py`, registered by the agent)
  recomputes everything derived and refuses what would break later stages; the same editor
  checks drafts live (`POST …/plan/check`, `…/report/check`). `write_file` rewrites a file an
  agent declares editable (`Agent.editable_files`; the Coder's `experiment.py`).
* Every decision is bound to what the page showed (`payload.expect`: the checkpoint's nonce and
  artifact versions), so a decision made on a stale page is refused, not applied to something
  you did not see. Queued actions can be undone until a worker picks them up.
* Drafts are saved as you type in `<run>/ui/drafts/` (outside everything the engine reads);
  sending an edit removes its draft, undoing it restores the draft.
* Live updates: the server tails `events.jsonl` by byte offset and streams new events (SSE);
  the page refreshes what they touch.

## Security

The server runs on shared machines and is reached through an ssh tunnel.

* A random token, printed at start-up (and written to `--token-file`, mode 0600), is traded once
  for an HttpOnly, SameSite=Strict session cookie named after the port; the token never sits in
  a cookie or a URL query.
* Only loopback `Host` names are accepted (DNS rebinding); state-changing calls need
  `X-Requested-With: rtool` and a loopback `Origin` (cross-site requests); request bodies are
  capped (paper uploads 64 MB, the rest 8 MB).
* Run and project names must be plain names; every file served or written is resolved inside its
  folder (symlinks included). Paper web pages and SVG figures are served with a sandbox policy;
  the app shell with a strict Content Security Policy.
* API keys are never shown or sent: the settings page says only whether each is present.

## Running it

```bash
rtool ui --mode local --demo --open        # a laptop: demo runs with simulated papers, model and cluster
```

On Barkla the UI runs in your own account (a small CPU job, or a process tied to your ssh
session) and is opened through a tunnel; see the [Barkla runbook](barkla_runbook.md).

## Development

```bash
.venv/bin/python -m research_tool.testing.demo_run runs/demo     # demo runs, one stopped at each checkpoint
.venv/bin/rtool ui --mode local --runs-root runs/demo --token-file runs/ui-token
cd frontend && npm ci && npm run dev                              # Vite on :5173, /api forwarded to :8765
```

The frontend is React + TypeScript (Vite), Tailwind, Radix primitives, TanStack Query, React
Router, CodeMirror 6 (code and Markdown editors, loaded on demand) and react-markdown. Stage pages
are loaded when first opened.

* API types: `scripts/build_frontend.sh` writes the OpenAPI schema and generates
  `src/api/schema.d.ts`; name the types you use in `src/api/types.ts`.
* Build: `scripts/build_frontend.sh` builds into `src/research_tool/web/static`. Commit the
  result: the cluster has no Node. A test fails when the committed build is older than the
  sources.
* Tests:
  * `npm test` (Vitest + Testing Library; CodeMirror is replaced by a text box),
    `npm run typecheck`, `npm run lint`;
  * `.venv/bin/pytest tests/web` for the API (security, every page's routes, actions through the
    engine on demo runs);
  * `npm run build && npm run e2e`: Playwright drives the built app against a real local server,
    takes a new question through every checkpoint, and checks each page for WCAG 2.1 A/AA
    problems with axe, in the light and the dark theme. Playwright's own Chromium does not run on
    macOS 13, so the installed Google Chrome is used, with a fresh temporary profile.
* Node: pinned to versions that run on Node 22.0 (Vite 6, Vitest 3). `npm audit` reports two
  moderate advisories in Vitest's mocker, which only runs in development.

## Adding a page for a new agent

1. Give the agent's outputs what a page needs: plain artifacts, `ctx.emit(...)` events for
   progress, and (if researchers may edit an output) an editor registered with
   `@artifact_editor("<artifact>")` or files in `editable_files`.
2. Add an API module in `src/research_tool/web/api/` that reads the run (`get_run`), returns
   page-shaped Pydantic models, and register it in `api/__init__.py`. Never write engine state:
   changes are actions.
3. Run `scripts/build_frontend.sh` to regenerate the types, add hooks in `src/api/hooks.ts`
   (query keys that start with `"artifact"` are refreshed by the event stream).
4. Write the page in `src/pages/<name>/`, using `CheckpointCard`, `FailureCard`, `ReopenButton`
   and the shared UI pieces, and add it to `PAGES` in `src/pages/StagePage.tsx` under the
   agent's name. Until then the generic page shows the stage.
5. Tests: an API test on a demo run (`tests/web/`), a page test (`src/pages/<name>/*.test.tsx`),
   and a step in the end-to-end test if the page has a checkpoint.
