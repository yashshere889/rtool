#!/usr/bin/env bash
# Discovery probe for Research Tool on Barkla (milestone M0).
#
# Run it on a Barkla LOGIN node (it only reads information and makes a few HTTPS HEAD requests):
#
#     bash probe_barkla.sh > ~/rtool-probe.txt 2>&1
#
# Optional: also test compute nodes by submitting two tiny jobs from your scratch space
# (1 CPU core for ~2 minutes on 'nodes', and 1 L40S GPU for at most 15 minutes on 'gpu-l40s'):
#
#     bash probe_barkla.sh --with-jobs > ~/rtool-probe.txt 2>&1
#     # later, once both jobs have finished (check: squeue -u $USER):
#     cat ~/scratch/rtool-probe/*.out
#
# --jobs-only submits just the two jobs. PROBE_GPU_PARTITION / PROBE_GPU_TYPE pick another
# GPU partition (default gpu-l40s / l40s). Container checks hide ~/.local packages and use a
# throwaway home, exactly as rtool runs the container.
#
# Nothing here installs, downloads models, or changes any configuration.

set -u
LIMIT=60
WITH_JOBS=0
LOGIN_CHECKS=1
[[ "${1:-}" == "--with-jobs" ]] && WITH_JOBS=1
[[ "${1:-}" == "--jobs-only" ]] && WITH_JOBS=1 && LOGIN_CHECKS=0
GPU_PARTITION="${PROBE_GPU_PARTITION:-gpu-l40s}"
GPU_TYPE="${PROBE_GPU_TYPE:-l40s}"
export PYTHONNOUSERSITE=1  # the container's Python must not import ~/.local packages
unset PYTHONPATH

section() { printf '\n===== %s =====\n' "$1"; }
run() {
  printf '$ %s\n' "$1"
  timeout 120 bash -lc "$1" 2>&1 | head -n "$LIMIT"
}

URLS=(
  https://api.openalex.org/works?per-page=1
  https://api.semanticscholar.org/graph/v1/paper/search?query=test\&limit=1
  https://export.arxiv.org/api/query?search_query=all:test\&max_results=1
  https://api.unpaywall.org/v2/10.1038/nature12373?email=probe@example.org
  https://api.crossref.org/works?rows=1
  https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-Nano-12B-v2
  https://pypi.org/simple/pip/
  https://registry-1.docker.io/v2/
  https://nvcr.io/v2/
  https://github.com
)
http_checks() {
  for url in "${URLS[@]}"; do
    printf '%-80s ' "$url"
    curl -s -o /dev/null -m 15 -w '%{http_code} %{time_total}s\n' "$url" || echo "FAILED (no connection)"
  done
}

if [[ "$LOGIN_CHECKS" -eq 1 ]]; then
section "host"
run 'hostname; whoami; date -Is; head -3 /etc/os-release; nproc; free -g | head -2'

section "storage and quotas"
run 'ls -ld ~/scratch ~/fastscratch ~/data ~/localscratch'
run '/opt/apps/user_tools/display_disk_usage_quotas.sh'
run 'lfs quota -u "$USER" /mnt/scratch; lfs quota -u "$USER" /mnt/fastscratch'
run 'quota -u "$USER" 2>&1 | head -12'
run 'mount | grep -E " type (lustre|nfs)" | awk "{print \$3, \$5, \$6}" | head -20'

section "modules of interest"
run 'module -t avail 2>&1 | grep -i -E "nemotron|vllm|apptainer|singularity|miniforge|anaconda|pixi|python|cuda|cudnn|pytorch|torch|tensorflow|pandoc|texlive|latex|java|rclone|git" | sort -u'
run 'module spider vllm 2>&1 | head -20'

section "nemotron module"
run 'module show nemotron/nano-12b-v2'
run 'ls -la /opt/apps/slurm_examples/gpu/nemotron/ && head -80 /opt/apps/slurm_examples/gpu/nemotron/*.sh'
run 'module load nemotron/nano-12b-v2 && echo "NEMOTRON_SIF=$NEMOTRON_SIF" && ls -la "$NEMOTRON_SIF" && type nemotron && file "$(command -v nemotron)" && head -60 "$(command -v nemotron)"'

section "what is inside the nemotron container (no GPU needed)"
run 'module load apptainer/1.3.6 nemotron/nano-12b-v2 && apptainer exec "$NEMOTRON_SIF" python3 - <<EOF
import importlib.util as u, sys
print(sys.version)
for m in ["vllm", "torch", "transformers", "mamba_ssm", "causal_conv1d", "xgrammar", "outlines", "flash_attn", "sentence_transformers"]:
    print(f"{m:24s} {bool(u.find_spec(m))}")
EOF'
run 'module load apptainer/1.3.6 nemotron/nano-12b-v2 && apptainer exec "$NEMOTRON_SIF" python3 -m pip list 2>/dev/null | grep -i -E "^(vllm|torch|transformers|mamba|causal|xgrammar|outlines|accelerate|flash|triton|openai)" '
run 'module load apptainer/1.3.6 && apptainer --version && echo "APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-<unset: defaults to ~/.apptainer/cache>}"'

section "slurm"
run 'sinfo --version; sinfo -o "%P %a %l %D %G" | head -40'
run 'sinfo -p gpu-l40s,gpu-h100,gpu-v100 -N -o "%N %P %G %T" '
run 'scontrol show partition gpu-l40s | head -25'
run 'sacct -X -n -S now-7days -o JobID,JobName%20,Partition,State,ExitCode | head -5; echo "sacct exit code: ${PIPESTATUS[0]}"'
run 'sacctmgr -n show assoc user="$USER" format=account%20,partition%20,qos%40 2>&1 | head'
run 'squeue -u "$USER"'
run 'squeue -h -u "$USER" -r -o "%i|%T|%r|%j" | head -5; echo "squeue exit code: ${PIPESTATUS[0]}"'
run 'sacct -X -n -P -S now-7days -o JobID,State,ExitCode,Elapsed | head -8; echo "sacct exit code: ${PIPESTATUS[0]}"'
run 'squeue -h -j 1 -o "%i|%T" 2>&1; echo "squeue on unknown job exit code: $?"'

section "python / sqlite (orchestrator runtime)"
run 'module load miniforge3/25.3.0-python3.12.10 && python - <<EOF
import sqlite3, sys
print(sys.version)
print("sqlite", sqlite3.sqlite_version)
con = sqlite3.connect(":memory:")
con.execute("create virtual table t using fts5(x)")
print("fts5 ok")
EOF'
run 'command -v pandoc latexmk pdflatex git rsync zstd pigz 2>&1'

section "internet from the LOGIN node (HTTP code, time)"
http_checks
fi

if [[ "$WITH_JOBS" -eq 1 ]]; then
  section "submitting compute-node probes"
  PROBE_DIR=~/scratch/rtool-probe
  mkdir -p "$PROBE_DIR" && cd "$PROBE_DIR" || exit 1
  declare -p URLS > urls.sh
  declare -f http_checks > http_checks.sh

  cat > cpu_probe.sh <<'EOF'
#!/bin/bash -l
#SBATCH -J rtool-probe-cpu
#SBATCH -p nodes
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:05:00
#SBATCH -o cpu_probe-%j.out
source ./urls.sh; source ./http_checks.sh
echo "== host"; hostname; date -Is
echo "== internet from a compute node"; http_checks
echo "== can a job submit another job? (nested sbatch, cancelled immediately)"
child=$(sbatch --parsable -p nodes -N 1 -n 1 -t 00:01:00 -J rtool-probe-child --wrap "sleep 5" 2>&1)
echo "child submission: $child"
[[ "$child" =~ ^[0-9]+$ ]] && scancel "$child" && echo "cancelled child $child"
echo "== squeue / sacct from a compute node"
squeue -u "$USER" | head -5
sacct -X -n -j "$SLURM_JOB_ID" -o JobID,State | head -3
EOF

  printf '%s\n' '#!/bin/bash -l' '#SBATCH -J rtool-probe-gpu' "#SBATCH -p $GPU_PARTITION" '#SBATCH -N 1' \
    "#SBATCH --gres=gpu:$GPU_TYPE:1" '#SBATCH -t 00:15:00' '#SBATCH -o gpu_probe-%j.out' > gpu_probe.sh
  cat >> gpu_probe.sh <<'EOF'
export PYTHONNOUSERSITE=1; unset PYTHONPATH
mkdir -p "$PWD/container-home"
echo "== host"; hostname; date -Is; nproc; free -g | head -2
echo "== allocation"; env | grep -E "^SLURM_(CPUS|JOB_CPUS|MEM|GPUS|TRES)" | sort
echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
nvidia-smi | head -15
echo "== local scratch"; df -h /tmp/users/$USER 2>/dev/null || df -h /tmp
module purge
module load apptainer/1.3.6 nemotron/nano-12b-v2
box() { apptainer exec --nv --home "$PWD/container-home" "$NEMOTRON_SIF" "$@"; }
echo "== torch inside the nemotron container"
box python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "== bf16 matmul on the GPU"
box python3 -c "import torch; x=torch.randn(512,512,device='cuda',dtype=torch.bfloat16); print('bf16 ok', (x@x).float().abs().mean().item() > 0)" 2>&1 | tail -1
echo "== vllm inside the nemotron container"
box python3 -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -1
echo "== localhost port binding (for a co-located vLLM server)"
box python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print('bound port', s.getsockname()[1])"
EOF

  for job in ${PROBE_JOBS:-cpu gpu}; do
    echo "$job probe: $(sbatch --parsable "${job}_probe.sh")"
  done
  echo "results will appear in $PROBE_DIR/*.out (check with: squeue -u \$USER)"
fi

section "done"
echo "Send the full output of this script back (it contains no secrets)."
