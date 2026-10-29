# Research Tool (`rtool`)

A resumable, human-gated multi-agent research pipeline for the University of
Liverpool's **Barkla** HPC cluster, powered only by a local LLM (NVIDIA
Nemotron Nano 12B v2, served by vLLM on the cluster's GPUs).

```
literature ─▶[review]─▶ hypotheses ─▶[pick 1 of 3]─▶ plan ─▶[approve]─▶ fetch data
  ─▶ code: write ─▶ smoke test ─▶[approve GPU runs]─▶ Slurm runs ─▶ analysis ─▶ report ─▶ review ─▶[approve]
```

| agent | what it does |
|---|---|
| **Literature** | writes down how it reads the topic (key concepts, what is out of scope), plans queries, searches OpenAlex / Semantic Scholar / arXiv, screens each paper against the key concepts, snowballs citations, downloads open-access papers (checking each copy is the paper), extracts findings with verified quotes and numbers, builds a knowledge map with quality checks and a SQLite knowledge base |
| **Hypothesis** | mines gaps and contradictions, drafts grounded hypotheses, checks novelty and feasibility, scores them, offers 3 |
| **Planner** | design, resources (checked on Hugging Face) and compute (fitted to Barkla's partitions and your budget) |
| **fetch** | downloads the plan's datasets and models, quota-checked |
| **Coder** | writes `experiment.py` from a domain-pack skeleton, smoke-tests it on CPU, repairs failures (playbook first, then the model), runs the full experiment as a Slurm array after your approval, retries failed runs, and computes every statistic and figure deterministically |
| **Writer** | writes the report: facts sections by code, prose by the model with results as placeholders filled from the data and citations limited to the corpus |
| **Reviewer** | automated checks, an audit of `experiment.py` against the plan (conditions, held-out data, metric definitions, repairs; majority of three), three independent rubric reviews (median), edits the text, sends major experiment problems back as change requests |

## Status

- **Built and tested** (230 tests, mock LLM + fake Slurm, including the whole
  pipeline end to end with the experiment code really executed): all six agents,
  the engine, gates, change requests, Slurm episodes, preemption handling, archives.
- **On Barkla** (2026-10-08): environment, weights and `rtool doctor` green;
  `rtool llm-bench` on L40S/A40 (structured output 100 % after repair, 60k-token
  recall, ~500 tokens/s at 32 parallel requests). Still to run there: the
  `smoke` pipeline round trip and a first real topic.

## Using it on Barkla

```bash
scripts/sync_to_barkla.sh                                  # on the Mac: code -> ~/"Research tool"
ssh barkla 'bash -l ~/"Research tool"/scripts/setup_barkla_env.sh'
```

On Barkla:

```bash
module load miniforge3/25.3.0-python3.12.10
source /mnt/fastscratch/users/$USER/rtool/venv/bin/activate
export RTOOL_CONFIG="$HOME/Research tool/configs/barkla.yaml"

rtool doctor
rtool new sleep --topic "closed-loop auditory stimulation and memory" --pack generic
rtool run sleep                      # submits an episode job; it stops at the first gate
rtool status sleep                   # stages, gates, files to read
rtool approve sleep                  # or: rtool reject sleep -f "focus on older adults"
rtool select sleep 1                 # at the hypothesis gate
rtool reopen sleep --stage hypotheses   # change your mind later (later stages redo only if it changes)
rtool edit sleep experiment_plan --from plan.json   # replace an output with your edited version
rtool list                           # every run and what it waits for
rtool stats sleep                    # where its wall time went: queue waits, gates, stages, vLLM starts
rtool stop sleep                     # cancel its episode job (resume with rtool run)
rtool archive sleep                  # when done: results, report and code to ~/data
```

Secrets (you create them, `chmod 600`): `~/.config/rtool/secrets.env` with
`OPENALEX_API_KEY`, `S2_API_KEY` and `UNPAYWALL_EMAIL`. Episodes load the file
themselves; the keys are never passed on to experiment code, the vLLM server or
Slurm jobs. Details, paths and
cluster facts: [docs/barkla_runbook.md](docs/barkla_runbook.md).

## Web UI

`rtool ui` serves a browser interface (the `web` extra): start research from a question, follow
every stage live, and decide at each checkpoint. Each stage has its own page:

- **Literature**: every paper with Open full paper (PDF or text with the verified findings
  marked) and Summary, Methodology, Key findings, Limitations, Gaps, Datasets, References and
  Cite; a map of topics, methods and datasets; gaps and open problems; full-text search; keep or
  drop papers; upload a PDF you have.
- **Hypotheses**: the three offered side by side with the papers they use, their citations
  ("show in paper") and references; choose one, edit it first, or change your choice later.
- **Experiment plan**: the whole plan, editable (with live checks and what applying redoes),
  datasets and models with links, compute against the budget, then the downloads.
- **Code & runs**: the code with `experiment.py` editable (tested again before any GPU time), a
  diff with the generated version, smoke tests, repairs, every run with its log, the results.
- **Report** and **Review**: the report with linked citations, editable prose (placeholders and
  citations as chips), print to PDF; the review's verdict, scores, checks, issues and edits.

```bash
rtool ui --mode local --demo --open      # try it on a laptop: demo runs, simulated papers, model and cluster
```

On Barkla the UI runs as a small CPU job in your own account and listens on a Unix socket only
you can open. From the Mac (University VPN on), one command starts it (or finds the running one),
opens an ssh tunnel and the browser, already signed in:

```bash
scripts/rtool-ui                         # Ctrl-C closes the tunnel; the job keeps running
scripts/rtool-ui stop                    # end the UI job (runs keep going without it)
```

Details, and the by-hand route, in the [Barkla runbook](docs/barkla_runbook.md#the-web-ui).

How it works, security, development, tests (unit, API, end-to-end with accessibility checks)
and how to add a page for a new agent: [docs/frontend.md](docs/frontend.md).

## How it works

A *run* is a directory. Each stage is an **agent** that reads typed, versioned
**artifacts** and writes new ones; agents never call each other. An **episode**
(a Slurm job) applies your decisions, runs every ready stage and stops when the
run needs you, a Slurm job, a GPU, or more time — nothing stays running between
episodes, and a killed episode resumes where it stopped. More in
[docs/architecture.md](docs/architecture.md).

## Extending

- **New agent:** one class + one line of pipeline YAML, no engine changes:
  [docs/adding_an_agent.md](docs/adding_an_agent.md).
- **New field:** a domain pack in `src/research_tool/packs/<name>/` (hints, code
  skeleton, prompt overrides).
- **Change the flow:** edit the pipeline YAML; set any gate to `auto`.

## Development (on the Mac; nothing is downloaded)

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/rtool new demo --topic "toy topic" --pipeline toy --runs-root ./runs
.venv/bin/rtool run runs/demo/<run-id> --simulate-jobs
```
