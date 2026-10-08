# Barkla runbook

What `scripts/probe_barkla.sh` found on 2026-10-08, and how rtool is set up on Barkla.

## Where things live

| What | Path | Why there |
|---|---|---|
| Code (synced from the Mac) | `~/Research tool` | small; home is backed up |
| Python environment | `/mnt/fastscratch/users/$USER/rtool/venv` (+ `pip-cache/`) | home quota is 150k files; ~8k files |
| Runs (one folder per run) | `/mnt/scratch/users/$USER/rtool/projects` | jobs must be submitted from scratch |
| Datasets / models for experiments | `/mnt/scratch/users/$USER/rtool/resources` | shared by runs |
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
Experiments stay on the common partitions (gpu-l40s, gpu-h100, gpu-v100).
