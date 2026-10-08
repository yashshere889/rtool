#!/bin/bash -l
# Create or refresh the rtool Python environment on Barkla. Run on a login or viz node.
#
#     bash -l ~/"Research tool"/scripts/setup_barkla_env.sh [--fresh]
#
# The venv lives in fastscratch: the Barkla docs ask users not to install Python
# environments in $HOME (75 GB / 150k-file quota). Only small pure-Python
# dependencies are installed here; vLLM and PyTorch live in Apptainer images.
#
# The environment is self-contained: it ignores ~/.local packages, any pip config and
# pip's cache in your home (it keeps its own next to the venv). --fresh rebuilds it.
set -euo pipefail
SRC="${RTOOL_SRC:-$HOME/Research tool}"
ENV_DIR="${RTOOL_ENV:-/mnt/fastscratch/users/$USER/rtool/venv}"
PYTHON_MODULE="${RTOOL_PYTHON_MODULE:-miniforge3/25.3.0-python3.12.10}"

export PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null PIP_CACHE_DIR="$(dirname "$ENV_DIR")/pip-cache"
unset PYTHONPATH PYTHONHOME

module purge
module load "$PYTHON_MODULE"
mkdir -p "$(dirname "$ENV_DIR")"
if [[ "${1:-}" == "--fresh" && -d "$ENV_DIR" ]]; then
  rm -rf "$ENV_DIR"
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  python3 -m venv "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install --quiet --upgrade pip
"$ENV_DIR/bin/python" -m pip install --quiet -e "$SRC[dev,barkla]"
"$ENV_DIR/bin/rtool" --version
echo "files in venv: $(find "$ENV_DIR" -type f | wc -l) (counts toward the fastscratch 500k-file quota)"
echo "use it with:  module load $PYTHON_MODULE && source $ENV_DIR/bin/activate"
