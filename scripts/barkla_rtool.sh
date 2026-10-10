#!/bin/bash -l
# Run rtool on a Barkla login node with the project's own environment, e.g. from the Mac:
#
#     ssh barkla 'bash -l ~/"Research tool"/scripts/barkla_rtool.sh ui --status'
#
# Same isolation as the jobs: the venv in fastscratch, no ~/.local packages, no PYTHONPATH,
# and configs/barkla.yaml unless RTOOL_CONFIG says otherwise.
set -euo pipefail
ENV_DIR="${RTOOL_ENV:-/mnt/fastscratch/users/$USER/rtool/venv}"
PYTHON_MODULE="${RTOOL_PYTHON_MODULE:-miniforge3/25.3.0-python3.12.10}"
export PYTHONNOUSERSITE=1 RTOOL_CONFIG="${RTOOL_CONFIG:-$HOME/Research tool/configs/barkla.yaml}"
unset PYTHONPATH PYTHONHOME
module purge >/dev/null 2>&1
module load "$PYTHON_MODULE" >/dev/null 2>&1
exec "$ENV_DIR/bin/rtool" "$@"
