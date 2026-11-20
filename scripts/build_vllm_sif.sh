#!/bin/bash -l
# Build an Apptainer image with vLLM. Needed for any model the nemotron module's image (vLLM 0.11.0)
# is too old for: Qwen3.6-35B-A3B (the default) needs vLLM >= 0.19. Run on a viz node — the Barkla docs ask for
# lengthy builds and data transfers to happen on barklaviz1/2, not the login node.
#
#     bash build_vllm_sif.sh [vllm-image-tag]
#
# Pulls the official vllm/vllm-openai image from Docker Hub (several GB) and writes
# a single .sif file (one inode). The Qwen3.6 model card needs vLLM >= 0.19.
set -euo pipefail
TAG="${1:-v0.19.1}"
OUT_DIR="${RTOOL_CONTAINERS:-/mnt/scratch/users/$USER/rtool/containers}"
# Keep Apptainer's cache and temp space out of $HOME (75 GB / 100k-file quota).
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/mnt/scratch/users/$USER/.apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/users/$USER/apptainer-tmp}"
mkdir -p "$OUT_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

module purge
module load apptainer/1.3.6
apptainer build "$OUT_DIR/vllm-$TAG.sif" "docker://vllm/vllm-openai:$TAG"
apptainer exec "$OUT_DIR/vllm-$TAG.sif" python3 -c "import vllm; print('vllm', vllm.__version__)"
echo "built $OUT_DIR/vllm-$TAG.sif — set llm.server.container to this path"
