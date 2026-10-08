#!/bin/bash -l
# Download the Nemotron weights once into the Hugging Face cache that vLLM reads
# (llm.server.hf_home). Runs as a 1-core job on the CPU partition: compute nodes have
# internet, and Barkla only accepts jobs submitted from scratch or fastscratch:
#
#     mkdir -p ~/scratch/rtool/logs && cd ~/scratch/rtool/logs
#     sbatch ~/"Research tool"/scripts/fetch_model.sh
#
# Safe to re-run: files already downloaded are skipped.
#SBATCH -J rtool-fetch-model
#SBATCH -p nodes
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o rtool-fetch-model-%j.out
set -euo pipefail
MODEL="${RTOOL_MODEL:-nvidia/NVIDIA-Nemotron-Nano-12B-v2}"
export HF_HOME="${RTOOL_HF_HOME:-/mnt/scratch/users/$USER/rtool/hf}"
ENV_DIR="${RTOOL_ENV:-/mnt/fastscratch/users/$USER/rtool/venv}"

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
module purge
module load miniforge3/25.3.0-python3.12.10
mkdir -p "$HF_HOME"
echo "downloading $MODEL into $HF_HOME on $(hostname) at $(date)"
"$ENV_DIR/bin/python" - "$MODEL" <<'EOF'
import sys

from huggingface_hub import snapshot_download

# weights, config, tokenizer and the model's own code (needed by --trust-remote-code); skip images
path = snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.py", "*.safetensors", "*.md"])
print("snapshot:", path)
EOF
du -sh "$HF_HOME/hub"
echo "done at $(date)"
