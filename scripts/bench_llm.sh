#!/bin/bash -l
# Measure the served Nemotron model (milestone M2): starts vLLM on one GPU, runs
# `rtool llm-bench`, and writes llm_profile-<job>.json next to the job log (~20-40 min).
# Submit from scratch; command-line flags override the partition below:
#
#     mkdir -p ~/scratch/rtool/bench && cd ~/scratch/rtool/bench
#     sbatch ~/"Research tool"/scripts/bench_llm.sh
#     sbatch -p gpu-h100 --gres=gpu:h100:1 ~/"Research tool"/scripts/bench_llm.sh
#
#SBATCH -J rtool-llm-bench
#SBATCH -p gpu-l40s
#SBATCH -N 1
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 01:30:00
#SBATCH -o rtool-llm-bench-%j.out
set -eo pipefail
SRC="${RTOOL_SRC:-$HOME/Research tool}"
ENV_DIR="${RTOOL_ENV:-/mnt/fastscratch/users/$USER/rtool/venv}"

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
module purge
module load apptainer/1.3.6 miniforge3/25.3.0-python3.12.10 nemotron/nano-12b-v2
source "$ENV_DIR/bin/activate"
export RTOOL_CONFIG="$SRC/configs/barkla.yaml"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
rtool llm-bench --serve --out "llm_profile-$SLURM_JOB_ID.json" "$@"
