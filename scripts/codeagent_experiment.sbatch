#!/bin/bash -l
# Serve Qwen3-Coder-Next with tool calling and run scripts/codeagent_experiment.py on one rtool run.
#
#   sbatch scripts/codeagent_experiment.sbatch <rtool run dir> <workspace dir>
#
# Submit from scratch (compute nodes can only write there). Same server settings as
# configs/barkla-qwen3-coder-next.yaml, plus the tool-call parser codeagent needs.
#SBATCH --job-name=codeagent-exp
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --time=03:00:00
#SBATCH --output=codeagent_exp_%j.log
#SBATCH --export=NONE

set -uo pipefail
RUN_DIR="${1:?run dir}"; WS="${2:?workspace dir}"
R=/mnt/scratch/users/$USER/rtool; F=/mnt/fastscratch/users/$USER/rtool
MODEL=Qwen/Qwen3-Coder-Next-FP8

module purge; module load apptainer/1.3.6
# Isolation from the old project's ~/.local and caches (memory: barkla-fresh-isolated-project).
export PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null HF_HOME=$R/hf; unset PYTHONPATH
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24

mkdir -p "$WS"
apptainer exec --nv --home $R/vllm-home --bind /mnt/scratch,/mnt/fastscratch,/tmp \
    --env HF_HOME=$HF_HOME --env VLLM_USE_FLASHINFER_MOE_FP8=0 --env VLLM_USE_DEEP_GEMM=0 \
    $R/containers/vllm-v0.15.0.sif \
    vllm serve $MODEL --host 127.0.0.1 --port 8000 --tensor-parallel-size 2 \
        --max-model-len 131072 --gpu-memory-utilization 0.90 --safetensors-load-strategy eager \
        --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    > "$WS/vllm.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; sleep 5; kill -9 $SERVER 2>/dev/null' EXIT

for _ in $(seq 1 180); do
    curl -sf http://127.0.0.1:8000/health >/dev/null && break
    kill -0 $SERVER 2>/dev/null || { echo "vLLM died during startup"; tail -40 "$WS/vllm.log"; exit 1; }
    sleep 10
done
curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "vLLM not up after 30 min"; exit 1; }
echo "server up after ${SECONDS}s; serving: $(curl -s http://127.0.0.1:8000/v1/models)"

export PATH=$F/codeagent-venv/bin:$PATH   # the agent's `python` is this venv too
cd "$WS" && python $F/codeagent-driver/codeagent_experiment.py "$RUN_DIR" "$WS" --model $MODEL
echo "driver exit $? at $(date)"
