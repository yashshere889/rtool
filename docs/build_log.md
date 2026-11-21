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

---

## Stage 1 — analyse existing data (local gate passed 2026-11-21)

Built:
- **Units mode** (`replicates: units` in a pack):
  - A data analysis runs once per condition. Its `run()` returns `{"units": [{"unit", <metric>,
    "batch"?}, ...]}`, one entry per donor or sample. The harness checks the shape and stores the
    per-unit values and their means.
  - `collect` expands the entries into per-unit records. `results.aggregate_units` compares conditions
    across units: unpaired (Welch, Mann–Whitney), or paired by unit id when the same units appear in both.
  - `confounding_issues` reports a condition that shares no batch with the baseline, as `results.issues`
    and a results note.
  - The planner asks for `PlanCoreUnits` (with `unit_of_analysis`), sets one seed, skips the seed checks,
    and uses a units-mode prompt.
  - The Coder gets the units contract. Writer, Reviewer, report placeholders and tables say "donors",
    not "seeds".
- **Packs:**
  - `omics`: pseudobulk hints, GEO, CELLxGENE, url, local and pip sources, PubMed and Europe PMC first,
    and four review questions.
  - `tabular`: surveys, registries and other tables.
  - Both use `packs/units.py:pseudoreplication_issues`, which flags a unit of analysis that repeats within
    a donor or sample (cells, reads, visits...).
- **Pack hooks:** `literature_sources` (validated against the Literature agent's sources) and
  `review_checks` (the questions go into the review prompt).
- **Sources:** `PubMedSource` (E-utilities: ESearch JSON, then EFetch XML) and `EuropePMCSource` (REST
  search, `resultType=core`), plus GEO (series folders on NCBI's HTTPS FTP) and CELLxGENE (newest
  dataset version, its H5AD asset) as `geo` and `cellxgene` resource sources. The planner gets
  per-source rules for packs that limit their sources.
- **Validity:**
  - Units packs need no seeds.
  - The scorer counts units in L1 and L2, and adds the L2 check "the data show no design problem".
  - Two omics cases, `pseudobulk-planted` (fold 2 and fold 1.5) and `pseudobulk-null` (4 seeds). Both use
    synthetic single-cell counts in which donors vary.

References (standard library, Mac, 2026-11-21): donor-level permutation tests:
- Planted fold 2 (6+6 donors): log2 difference 1.106, p 0.015.
- Planted fold 1.5 (10+10 donors): 0.630, p 0.013.
- Null seeds 1, 2, 4 and 5: p 0.45, 0.22, 0.39 and 0.64.

Seeds 3 and 6 were left out: their donor-level tests are significant by chance (p 0.022, 0.011), so they
would fault a correct tool. Treating cells as independent gives z = −3.2, −12.0, 7.2 and 3.9 on the
nulls, so a run that pseudoreplicates makes false claims.

Checked against the services' own documentation, read in the browser:
- Europe PMC search: endpoint, `resultType`, `pageSize` ≤ 1000, `cursorMark`, and the
  `resultList.result[]` fields used.
- NCBI E-utilities: ESearch and EFetch base URLs, `retmax` ≤ 10,000, `retmode=json` for ESearch, XML for
  PubMed EFetch, `tool`/`email`/`api_key`, and 3 requests per second without a key.
- GEO README: the `nnn` bucket rule and the `suppl/` and `matrix/` folders.
- CELLxGENE OpenAPI spec: `/v1/datasets/{id}/versions`, and `assets` with `filetype`, `url` and `filesize`.

From memory, not checked: the ESearch JSON keys (`esearchresult.idlist`), the PubMed XML element
names (they follow PubMed's DTD), Europe PMC's `PUB_YEAR:[a TO b]` syntax, and the HTML of NCBI's
folder listings. The live probes in `tests/barkla/test_live_sources.py` check all of these.

Proved by:
- `tests/integration/test_units_pipeline.py`: the shipped `pseudobulk-planted` case through the real
  engine with the demo model. Its plan is a donor-level analysis with no issues, and its results are 12
  donor records. The Welch test gives the reference's difference, 1.106 (p 0.0105). The report's methods
  describe donors, and the run scores L3.
- `tests/unit/test_units_analysis.py`: harness contract, collection, paired and unpaired, confounding,
  pseudoreplication.
- `tests/unit/test_bio_sources.py`: the four clients on fixtures in the documented formats.
- Full suite exit 0, web tests exit 0, ruff clean. API types regenerated and the bundle rebuilt (only
  its stamp changed).

Waits for Barkla, or for you:
- `pytest -m barkla` (the live probes).
- Real runs of both omics cases.
- The roadmap's three published reanalyses, which need a domain expert to choose datasets and known findings.
- A real Coder writing a units-mode analysis (the demo model's analysis was written by hand).

Not done: the web UI shows unit records like seed runs (seed 0 for every unit); it needs no change to
work, but a units-aware results table would read better.

Files: `schemas/{plan,code}.py`, `packs/__init__.py`, `packs/units.py`, `packs/omics/**`, `packs/tabular/**`,
`agents/coder/{agent,results,sanity}.py`, `agents/coder/harness/run.py`, `agents/coder/prompts/implement.j2`,
`agents/planner/{agent,checks,edit}.py`, `agents/planner/prompts/{plan_core,resources}.j2`,
`agents/writer/agent.py`, `agents/reviewer/agent.py`, `agents/reviewer/prompts/review.j2`,
`agents/literature/{agent,toolkit}.py`, `tools/scholarly/sources.py`, `tools/resources/{omics,checks,fetch}.py`,
`agents/tasks/fetch_resources.py`, `tools/report/render.py`, `validity/{cases,score}.py`, `benchmarks/omics/**`,
`docs/architecture.md`, `frontend/src/api/*`, `web/static/build-stamp.json`, `tests/unit/test_units_analysis.py`,
`tests/unit/test_bio_sources.py`, `tests/integration/test_units_pipeline.py`, `tests/barkla/test_live_sources.py`.

---

## Stage 2 — simulate or model (local gate passed 2026-11-21)

Built:
- **`biomodels` resource source** (`tools/resources/models.py`):
  - checked with `GET /model/files/{id}?format=json` (main files and sizes);
  - fetched with `GET /model/download/{id}?filename=…`;
  - every downloaded `.xml` must be SBML with a `<model>`, or the fetch fails and says why. That is a
    well-formedness check; full libSBML validation is not built.
- **Pack `result_checks`** (`"module:function"`, `(plan, records) -> issues`): run on finished
  experiments; their findings go into `results.issues` and the notes, so they fail L2 as `design_problem`.
- **`sysbio` pack:**
  - hints: the seed must matter (sampled parameter uncertainty or stochastic simulation), stiff
    solvers with rtol ≤ 1e-8, and report `mass_balance_error`;
  - sources: BioModels, url, local, pip;
  - literature: PubMed first;
  - four review questions;
  - result check `conservation_issues`: a conservation error above 1e-6 relative is flagged. That is
    also how a too-large step or a non-stiff solver usually shows; a separate stiffness check is not built.
- **Fixed in the sanity checks:** a lower-is-better error that is exactly zero everywhere no longer
  counts as a fake experiment. A conserving simulation's mass balance is 0 in every run, and the old rule
  failed it at L1.
- Three epidemic cases (`sysbio`), with references in the standard library (Mac, 2026-11-21):
  - `sir-final-size`: 0.792 over 8 seeds, against 0.800 from the final-size equation (R0 2).
  - `vaccination-shrinks-epidemic`: 0.792 against 0.330, paired t −11.3.
  - `ssa-method-null`: Gillespie's direct method against first reaction, paired t −1.04, 0.01 and 1.15
    at R0 1.5, 2 and 3.

Checked against BioModels' own API spec (biomodels.org/docs/biomodels-jummp-swagger.json): the
files and download endpoints, `ModelFiles.main[].name` and `fileSize`, and the `filename` parameter.

Proved by:
- `tests/integration/test_sysbio_pipeline.py`: a planted model case through the real engine. A mock
  BioModels server serves an SBML model, the fetch stage downloads and checks it, and the simulation
  reads it. The conserving version scores L3; a version that leaks mass is flagged
  (`results.issues`, a design-problem note) and scores L1, `design_problem`.
- `tests/unit/test_models_sources.py`: BioModels check, fetch and SBML problems, the conservation check,
  and the zero-error rule.
- Every shipped pack's cases load and generate their data (`test_validity_card.py`).
- Full suite exit 0, web tests exit 0, ruff clean. API types regenerated and the bundle rebuilt (only
  its stamp changed).

Waits for Barkla, or an expert:
- `pytest -m barkla` (the BioModels probe fetches BIOMD0000000012).
- Real runs of the three cases.
- The roadmap's "curated models reproduce their published outputs" and "known knockouts predicted
  correctly". Both need someone to choose models and published values, and to run libroadrunner or
  cobra on Barkla. Physics models are covered only by the generic pack's simulations (the queue cases).

Files: `tools/resources/{models,checks,fetch}.py`, `schemas/plan.py`, `packs/__init__.py`, `packs/sysbio/**`,
`agents/coder/{agent,sanity}.py`, `agents/planner/agent.py`, `testing/samples.py`, `benchmarks/sysbio/**`,
`frontend/src/api/*`, `web/static/build-stamp.json`, `tests/unit/test_models_sources.py`,
`tests/integration/test_sysbio_pipeline.py`, `tests/unit/test_validity_card.py`, `tests/barkla/test_live_sources.py`.

---

## Stage 3 — literature synthesis without compute (local gate passed 2026-11-21)

Built:
- **Pipeline `synthesis`** (literature → synthesis, gates `literature_review` and `final_report`).
  Run with `rtool new … --pipeline synthesis`.
- **Synthesis agent** (`agents/synthesis/`):
  - For each topic of the knowledge map, the model writes a paragraph sentence by sentence. Every
    sentence names the verified findings it rests on (C1, C2, …).
  - Code keeps a sentence only if every finding it names was offered and is verified (its quote was
    found in the paper), and every number it states appears in those findings. Dropped sentences are
    kept with their reasons.
  - It writes the review with `[@key]` citations, an Evidence section quoting each finding, and
    references. Outputs: `synthesis` (sentences and their findings) and `final_report` (the text, for the UI).
- **Validity:**
  - A `synthesis` case kind, scored against a labelled set (`expect.findings`: a paper-title fragment,
    or none for any paper, and terms; `min_recall`).
  - Cases gain `pipeline:`, the pipeline a case runs (default: `synthesis` for synthesis cases, else
    `research`).
  - The scorer re-checks the review itself rather than trusting the agent's filter. L1: every sentence
    rests on listed findings. L2: numbers and citations match those findings. L3: the known findings
    are stated.
- **`medicine` pack** (PubMed and Europe PMC first, three review questions), and the shipped case
  `statins-ldl-review` (statins lower LDL, stated by any paper).

Proved by `tests/integration/test_synthesis_pipeline.py`:
- The literature stage runs for real on the demo's fictional papers (PDFs, quote checks), then the
  synthesis stage. The stubbed model restated 19 findings and added an invented number and a citation
  to a finding nobody offered, in each of the 2 paragraphs. Those 4 sentences were dropped, and none of
  their text reached the review.
- The run scores L3 with recall 1.0 on two labelled demo findings. A third, false labelled finding
  gives `missed_findings` (recall 2/3). A sentence edited to carry numbers its quote lacks gives
  `unfaithful_review` at L1.
- Full suite exit 0, web tests exit 0, ruff clean, API schema unchanged.

Waits for Barkla, or an expert:
- A real model writing reviews.
- The statins case on real PubMed and Europe PMC results.
- The roadmap's exit test needs a labelled set from a domain expert (claim-support and citation
  accuracy with a threshold set beforehand). The scorer measures recall of known findings, and the
  dropped share is reported as a fact.

Not done: no LLM Reviewer for reviews (the Reviewer agent expects an experiment); the final checkpoint
shows the synthesis checks instead. The web report page has no review artifact to show for this pipeline.

Files: `agents/synthesis/**`, `pipelines/synthesis.yaml`, `core/registry.py`, `validity/{cases,runs,score}.py`,
`packs/medicine/pack.yaml`, `benchmarks/medicine/**`, `tests/integration/test_synthesis_pipeline.py`,
`tests/unit/test_validity_cases.py`, `tests/unit/test_validity_card.py`.

---

## Stage 4 — study design with a person in the loop (local gate passed 2026-11-21)

Built:
- **Pipeline `study`**: literature → hypotheses → protocol (gate `protocol_review`) → collect (in-stage
  gate `data_upload`) → analyse → report → review (gate `final_report`).
- **Protocol agent** (`agents/study/protocol.py`):
  - One model call designs the study (conditions, unit of analysis, primary outcome, the smallest
    effect worth detecting, materials, blinding).
  - Computed, not asked: the sample size (normal approximation with a small-sample correction; alpha
    split across the comparisons with the control), a seeded block-randomised allocation (paired: a
    random order per unit), and the data format.
  - Checklist: a control, a primary outcome fixed before data, a computed sample size, an independent
    unit, materials, blinding. Its failures become plan issues.
- **Collect** (`collect_data`): waits at `data_upload` until `data.csv` is uploaded (`rtool add-data`),
  or finds it among the run's local data (as validity cases provide it). It checks the file against the
  protocol (header, known conditions, numeric metrics, units unique per condition, at least two per
  condition). A file that fails stops the stage with each problem listed; fix it, upload again, `rtool retry`.
- **Analyse** (`analyse_table`): rtool writes its own analysis script (`experiment.py`, so it can be
  reread, rerun and audited) and executes it. The results are analysed like any data analysis (units
  mode, confounding and pack checks), so Writer, Reviewer and scorer work unchanged.
- **`rtool add-data <run> data.csv`**: copies the file and approves the open data checkpoint.
- **`study` pack**: four review questions (randomisation, blinding, attrition, analysis fixed in
  advance). Two shipped cases: `caffeine-reaction-time` (planted) and `placebo-null`.
- Sample-size check: 64 per group for d = 0.5, 34 paired, 26 for d = 0.8, and 78 with two comparisons.
  These are the textbook values.

References (standard library, Mac, 2026-11-21), Welch t:
- Planted trials: −5.06 (seed 1), −2.95 (seed 2).
- Null trials: −0.13, 0.09, 0.77, 1.07 (seeds 2–5). Seed 1 (−1.89) is left out as too close to the threshold.

Proved by `tests/integration/test_study_pipeline.py`:
- The shipped planted trial through the real engine. The protocol computes 45 per arm for d = 0.6 and
  a 90-row allocation in blocks of two, with every checklist item passing. The uploaded 90 participants
  are analysed (Welch, difference −39.8 ms, the reference's figure), and the run scores L3.
- A second run waits at `data_upload`. A broken upload (an unknown condition, a non-number) is refused
  with both problems named. After the correct file and `rtool retry`, the run finishes.
- Full suite exit 0, web tests exit 0, ruff clean, API schema unchanged.

Waits for Barkla, or you:
- A real model writing protocols.
- Running a real study (people or lab work).
- The exit test's "uploaded data of a defined format flows through without manual repair" is shown
  locally; a real lab's file will be the true test.

Not done: the web UI has no upload button (the CLI does it) and no protocol view (`protocol.md` is in the run).

Files: `agents/study/**`, `pipelines/study.yaml`, `packs/study/pack.yaml`, `core/registry.py`,
`cli/main.py` (add-data), `validity/runs.py` (data_upload in the auto gates), `benchmarks/study/**`,
`tests/integration/test_study_pipeline.py`.

---

## Stage 5 — GPU-heavy fields (local gate passed 2026-11-21)

Built:
- **Per-pack images**: site config `experiments.pack_images: {pack: {container, packages}}`. The Coder
  runs in the pack's own image, shows the Coder that image's packages, and records it in the code bundle.
- **Sources** `uniprot` (FASTA), `alphafold` (AlphaFold DB model, mmCIF) and `pdb` (RCSB mmCIF, checked
  through its header file). UniProt URLs and status codes, AlphaFold's prediction fields, and RCSB's
  download and header URLs were checked against their docs. AlphaFold's `/api` prefix is from memory.
- **Packs:**
  - `protein`: result check `leakage_issues`, which flags test sequences that are also in the training set.
  - `imaging`: plan check for held-out data.
  - `docking`.
  - `md`: result check `energy_drift_issues`, which flags a constant-energy check whose energy drifts
    more than 1 %.
  - Each has hints, sources and three review questions.
- **Case** `protein/motif-classifier` (planted, CPU). Reference on the Mac: length 0.482 against
  dipeptides 0.742, paired t 27.7.

Proved by `tests/integration/test_protein_pipeline.py`:
- The Coder chose the pack image `/images/protein.sif`, and its prompt lists that image's packages.
- A UniProt record was fetched through the real fetch stage from a mock server.
- The clean run scores L3; a split leaking 3 test sequences into training is flagged as a design
  problem and scores L1.
- `tests/unit/test_gpu_packs.py` covers the sources, both checks and per-pack images.
- Full suite exit 0, web tests exit 0, ruff clean. API types regenerated and the bundle rebuilt.

Waits for Barkla: pack images, GPU runs of real protein, imaging, docking and MD cases, and the
UniProt, AlphaFold and PDB probes. Not built: probes for the three new sources, cases for imaging,
docking and MD, and an SMILES or structure validator (those need RDKit or OpenMM).

Files: `core/config.py`, `agents/coder/agent.py`, `tools/resources/{structures,checks,fetch}.py`,
`schemas/plan.py`, `agents/planner/agent.py`, `packs/{protein,imaging,docking,md}/**`, `testing/samples.py`,
`benchmarks/protein/**`, `frontend/src/api/*`, `web/static/build-stamp.json`, `tests/unit/test_gpu_packs.py`,
`tests/integration/test_protein_pipeline.py`.

---

## Rules for every stage — what rtool refuses (local gate passed 2026-11-21)

The roadmap's rules for every stage say rtool refuses pathogen enhancement, toxin or toxic-protein
design, and controlled-access data. Stages 1–5 shipped the packs these rules exist for (omics, medicine,
protein, docking) without them. This adds them, for every pack, built-in or installed.

Built (`packs/refusals.py`):
- Default rules, in every pack:
  - work that makes a pathogen more dangerous: "gain of function", or raising transmissibility,
    virulence, lethality or host range, with a pathogen named;
  - designing a toxin, a toxic protein, a nerve agent, or a chemical or biological weapon, unless the
    text is about neutralising, detecting, diagnosing, inhibiting or vaccinating against one;
  - controlled-access human data, by accession: dbGaP (`phs`, `pht`, `phv`, `phd`, `phg`, `pha`, from
    dbGaP's study submission guide) and EGA (`EGAS`, `EGAN`, `EGAX`, `EGAZ`, `EGAR`, `EGAC`, `EGAP`,
    `EGAD`, from EGA's metadata schema page; it gives no digit count, so any run of digits matches).
- Where they apply:
  - `create_run` (CLI, web UI, demo) refuses the brief: topic, questions, constraints, search queries
    and datasets. `create_case_run` also checks a case's hypothesis.
  - The planner and the protocol agent stop their stage (it needs a person) when the selected
    hypothesis or the design's objective is refused. It is not a plan issue someone could approve past.
  - The resource check reports a controlled-access resource (by accession, or a dbGaP or EGA URL), so the
    planner chooses again; the fetch refuses one.
- A pack may add rules (`refuse:` in pack.yaml: a reason, `all` patterns that must all match, an
  optional `unless`). The defaults always apply.

Limits: these are regular expressions over the text. They stop requests written in plain words, not
someone who rephrases to get past them. They also refuse some descriptive work, for example "which
mutations increase transmissibility" (rephrased as "associated with higher transmissibility", it passes).

Proved by:
- `tests/unit/test_refusals.py`:
  - 8 traps are refused and 9 near misses pass (among them "design antibodies that neutralise snake venom
    toxins", "gain-of-function mutations in KRAS", "do not use dbGaP data").
  - Controlled-access data is caught by accession, or as a resource by its URL. The resource check and
    the fetch both refuse it before any request is made.
  - A pack's own rules (with `unless`) are added to the defaults; a rule that does not compile, or has
    no reason, fails when the pack loads.
  - A refused brief creates no run, and a case with a refused hypothesis is refused.
  - In a demo-model run, the planner stops at a refused objective: the stage fails and waits for a person.
- `tests/integration/test_study_pipeline.py`: the protocol stage stops the same way.
- Full suite exit 0, web tests exit 0, ruff clean, API schema unchanged.

Waits for Barkla: a real model's plans checked against these rules. Not built: a check of generated
hypotheses before one is selected (the plan stage checks the selected one), and a classifier.

Files: `packs/refusals.py`, `packs/__init__.py`, `agents/common.py`, `agents/planner/agent.py`,
`agents/study/protocol.py`, `service/runs.py`, `validity/runs.py`, `tools/resources/{checks,fetch}.py`,
`docs/architecture.md`, `tests/unit/test_refusals.py`, `tests/integration/test_study_pipeline.py`.
