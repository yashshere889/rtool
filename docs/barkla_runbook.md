# Barkla runbook

What `scripts/probe_barkla.sh` found on 2026-10-08, and how rtool is set up on Barkla.

## Where things live

| What | Path | Why there |
|---|---|---|
| Code (synced from the Mac) | `~/Research tool` | small; home is backed up |
| Python environment | `/mnt/fastscratch/users/$USER/rtool/venv` (+ `pip-cache/`) | home quota is 150k files; ~8k files |
| Runs (one folder per run) | `/mnt/scratch/users/$USER/rtool/projects` | jobs must be submitted from scratch |
| Datasets / models for experiments | `/mnt/scratch/users/$USER/rtool/resources` | shared by runs |
| Web UI job (socket, token, log) | `/mnt/scratch/users/$USER/rtool/ui` (mode 0700) | jobs run from scratch; only you can open it |
| Nemotron weights (23 GB) | `/mnt/scratch/users/$USER/rtool/hf` | read offline by vLLM |
| vLLM's home inside the container | `/mnt/scratch/users/$USER/rtool/vllm-home` | caches; hides `~/.local` |
| Secrets (you create, `chmod 600`) | `~/.config/rtool/secrets.env` | never printed |

Nothing from other projects is used: the venv ignores `~/.local`, pip config and the
pip cache in home; containers run with `PYTHONNOUSERSITE=1` and their own home;
jobs unset `PYTHONPATH`. (`~/.local/lib/python3.10` has packages that would
otherwise be imported by the container's Python 3.10, e.g. a newer `openai`.)

## Setup (once)

```bash
scripts/sync_to_barkla.sh                                   # on the Mac, after every code change
ssh barkla 'bash -l ~/"Research tool"/scripts/setup_barkla_env.sh'      # --fresh rebuilds
ssh barkla 'mkdir -p ~/scratch/rtool/logs && cd ~/scratch/rtool/logs && sbatch ~/"Research tool"/scripts/fetch_model.sh'
```

On Barkla, for every session:

```bash
module load miniforge3/25.3.0-python3.12.10
source /mnt/fastscratch/users/$USER/rtool/venv/bin/activate
export RTOOL_CONFIG="$HOME/Research tool/configs/barkla.yaml"
rtool doctor
```

LLM benchmark (one GPU, ~30 min): `cd ~/scratch/rtool/bench && sbatch ~/"Research tool"/scripts/bench_llm.sh`

## Facts checked on the cluster

- **OS / Slurm:** Rocky Linux 9.7, Slurm 25.11.6. `sacct` works (login and compute
  nodes). Jobs can submit jobs (nested `sbatch` works from a compute node).
- **Internet from compute nodes:** yes (OpenAlex, arXiv, Unpaywall, Crossref,
  Hugging Face, PyPI, GitHub). Semantic Scholar answers 429 without an API key.
- **Nemotron module:** `nemotron/nano-12b-v2` sets `NEMOTRON_SIF` (9.4 GB image).
  Inside: Python 3.10, vLLM 0.11.0, torch 2.8.0+cu128, transformers 4.57.1,
  xgrammar 0.1.25, mamba_ssm, causal_conv1d. No separate vLLM image is needed.
- **GPU nodes:** driver 580.159.04 (CUDA 13.0). One GPU gets 8 CPUs on gpu05
  (V100) and 24 CPUs on gpu-l40s (`DefCpuPerGPU=24`); node-local `/tmp` ≈ 350 GB.
- **Python for the orchestrator:** `miniforge3/25.3.0-python3.12.10`, sqlite 3.49.2 with FTS5.
- **Not available:** pandoc, latexmk (a `texlive/2023` module exists).
- **Quotas:** home 75 GB / 150k files; scratch 2 TB / 300k; fastscratch 500 GB / 500k.
- **Partitions** (all GPU partitions use QoS `gpulimits`):

  | partition | GPUs | limit | preemptible |
  |---|---|---|---|
  | gpu-l40s | 4 nodes × 2 L40S 48 GB | 3 days | no (tier 1000) |
  | gpu-h100 | 3 nodes × 4 H100 80 GB | 3 days | no |
  | gpu-v100 | 1 node × 4 V100 16 GB | 3 days | no |
  | gpu-l40s-low | 4 nodes × 2 L40S | 1 day | yes (requeue) |
  | gpu-a-lowsmall | 6 × 2 A100, 4 × 2 A40 | 1 day | yes (requeue) |
  | gpu-a100-lowbig | 4 nodes × 3–4 A100 | 1 day | yes (requeue) |

- **Queues (2026-10-08 15:20):** every L40S and H100 GPU was busy, with 36 jobs
  waiting for gpu-l40s and 55 for gpu-h100; Slurm estimated ~33 h before a
  1×L40S job would start. The V100 node was idle (16 GB is too small for
  Nemotron 12B in BF16).
- **The Nemotron image sets `HF_HOME=/opt/huggingface_cache` itself**, and an
  image's own variables beat the host's. rtool therefore passes its settings to
  the container as `APPTAINERENV_*` variables, which win.
- **vLLM start-up** (one L40S or A40): weights load in ~21 s with
  `--safetensors-load-strategy eager` (213 s with the default memory-mapped
  loading, which reads Lustre at ~0.12 GB/s; plain sequential reads reach
  1.6 GB/s); torch.compile ~12 s; engine warm-up ~23 s. Ready ~1.5 min after
  the job starts. KV cache: 119k tokens, i.e. ~10 concurrent 64k-token requests.
- **Reasoning:** vLLM's `deepseek_r1` reasoning parser refuses this model (its
  tokenizer has no single `<think>`/`</think>` tokens), so the server runs
  without one; rtool splits reasoning from answers itself and produces JSON
  after reasoning in a second call (`llm.structured_with_reasoning: false`).

## LLM benchmark (`rtool llm-bench`, 2026-10-08)

| | A40 (job 10814564) | L40S (job 10814824) |
|---|---|---|
| JSON, reasoning off: valid first try / after repair | 26/30 / 30/30 | 25/30 / 30/30 |
| JSON, reasoning on (think, then format) | 10/10 / 9/10 | skipped |
| reasoning split from the answer, no tag leaks | yes | — |
| needle found at 8k / 32k / 60k tokens | yes / yes / yes | yes / — / — |
| tokens/s at 1 / 8 / 32 concurrent requests | 23 / 146 / 396 | — / 161 / 507 |

Every invalid first try ran into `max_tokens` (`finish_reason=length`): greedy
decoding repeating the same item or sentence. Since then the client retries such
a reply briefly re-prompted and sampled at temperature 0.6 / top_p 0.95 (the
model card's settings) instead of showing the loop back to the model. One
request alone decodes at ~23 tokens/s (a 12B model is memory-bandwidth bound),
so long chains of thought take minutes; bulk work runs 32 requests at a time.

## Which GPU an LLM episode gets

Your choice (2026-10-08): any free GPU with at least 40 GB. Before each GPU
episode, rtool asks Slurm (`sbatch --test-only`) when each entry of
`slurm.episode.gpu_choices` would start — gpu-l40s, gpu-h100, gpu-l40s-low,
gpu-a-lowsmall (A40, A100), gpu-a100-lowbig — and takes the soonest; an earlier
entry is kept if it starts within `prefer_within_minutes` (30) of the soonest.
The low-priority partitions are preemptible: Slurm then sends TERM, the episode
stops cleanly within the 180 s grace period, the job is requeued with the same
id and resumes the run (it reclaims its own lease). `rtool events <run>` shows
which partition each episode went to and when it was expected to start.
`rtool stats <run>` (or `--json`) adds up where the run's wall time went: each episode's
queue wait and run time, how long gates waited for you and then for an episode to apply
the decision, time per stage and literature phase, vLLM start-up time (node and GPU) and
how busy each GPU episode kept the model.
Experiments stay on the common partitions (gpu-l40s, gpu-h100, gpu-v100).

## The web UI

The UI server runs as a small CPU job in your own account (login nodes forbid resident
processes): partition `nodes`, one core, 8 h by default (`--hours`, up to the partition's
3 days), no requeue. It listens on a Unix socket in `~/scratch/rtool/ui/` (mode 0700), so
only your account can reach it; it decides, submits episode jobs and cancels them exactly like
the CLI.

On the Mac (University VPN on):

```bash
scripts/sync_to_barkla.sh && ssh barkla 'bash -l ~/"Research tool"/scripts/setup_barkla_env.sh'  # once, and after updates
scripts/rtool-ui                 # start or find the UI job, tunnel, open the browser (signed in)
scripts/rtool-ui status          # where it runs and until when
scripts/rtool-ui stop            # end it; runs keep going without it
```

Ctrl-C closes only the tunnel; run `scripts/rtool-ui` again to reconnect (same job, same
token). Options: `--hours`, `--port` (on the Mac, default 8780), `--bind cluster`.

The launcher reads the compute node's host key through the login node into
`~/.config/rtool/barkla_known_hosts` on the Mac (your `~/.ssh/known_hosts` is not touched)
and the token from the job's 0600 token file; the token reaches the browser only in the
address's `#fragment`. By hand, on a login node `rtool ui` prints the same route:

```bash
ssh -N -L 8765:/mnt/scratch/users/$USER/rtool/ui/ui.sock -J barkla $USER@<node>   # on the Mac
```

then open http://127.0.0.1:8765 and paste the token from `~/scratch/rtool/ui/token`.

**Which route works** was checked with `scripts/probe_ui_access.sh` (2026-10-10, job on
node041): ssh from the Mac to the job's node through the login node works once the node's
host key is known (the login node's ssh trusts nodes on first use, the Mac's did not);
forwarding to a Unix socket on the node works with the socket in scratch or in the node's
`/tmp`; plain TCP forwarding through the login node to the node's port works too. Any user can
ssh to any compute node, so a server on the node's 127.0.0.1 would be no more private than one
on the cluster network; the socket is. `--bind cluster` (TCP, token only) stays as a fallback
for sites without ssh to nodes, and `rtool ui --mode login` as a session-bound last resort.

**Checked end to end** (2026-10-10): UI job 10838291 on node006; the smoke pipeline driven
from the UI's API: Resume submitted a GPU episode from inside the UI job (nested sbatch,
gpu-l40s-low, started within a minute), the checkpoint was approved through the UI, the CPU
episode ran the stage's Slurm job, and the last GPU episode finished the run.
