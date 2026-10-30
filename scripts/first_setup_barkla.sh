#!/bin/bash -l
# First-time setup in a new Barkla account, from a copy of the code in ~/"Research tool":
#
#     bash -l ~/"Research tool"/scripts/first_setup_barkla.sh
#
# Builds the Python environment, asks for the paper-search keys (only if there is no secrets file
# yet), queues the download of the model weights, then runs `rtool doctor`. Safe to run again.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RTOOL_SRC="${RTOOL_SRC:-$(dirname "$HERE")}"
SECRETS="${RTOOL_SECRETS:-$HOME/.config/rtool/secrets.env}"
HF_DIR="${RTOOL_HF_HOME:-/mnt/scratch/users/$USER/rtool/hf}"
LOGS="${RTOOL_LOGS:-/mnt/scratch/users/$USER/rtool/logs}"
MODEL_DIR="$HF_DIR/hub/models--nvidia--NVIDIA-Nemotron-Nano-12B-v2"

echo "== 1 of 4: Python environment (a few minutes)"
bash -l "$HERE/setup_barkla_env.sh"

echo "== 2 of 4: keys for searching papers"
if [ -e "$SECRETS" ]; then
  echo "keeping $SECRETS"
elif [ -t 0 ]; then
  echo "Paste each value and press Enter. Nothing shows while you paste. Press Enter alone to skip one."
  read -rsp "OpenAlex API key (free, openalex.org): " openalex; echo
  read -rsp "Semantic Scholar API key (free): " s2; echo
  read -rp "Your email, for Unpaywall: " email
  mkdir -p -m 700 "$(dirname "$SECRETS")"
  (
    umask 077
    {
      [ -n "$openalex" ] && echo "OPENALEX_API_KEY=$openalex"
      [ -n "$s2" ] && echo "S2_API_KEY=$s2"
      [ -n "$email" ] && echo "UNPAYWALL_EMAIL=$email"
      true
    } >"$SECRETS"
  )
  echo "saved in $SECRETS, readable only by you"
else
  echo "no keyboard here, so no keys were saved; add them to $SECRETS (chmod 600), one KEY=value per line"
fi

echo "== 3 of 4: model weights (23 GB, as a job)"
if [ -n "$(find "$MODEL_DIR/snapshots" -name '*.safetensors' -print -quit 2>/dev/null)" ]; then
  echo "already in $HF_DIR"
elif squeue -h -u "$USER" -n rtool-fetch-model -o %i | grep -q .; then
  echo "the download job is already queued or running"
else
  mkdir -p "$LOGS"
  (cd "$LOGS" && sbatch "$HERE/fetch_model.sh")
  echo "follow it with: squeue -u $USER; its log is in $LOGS"
fi

echo "== 4 of 4: rtool doctor"
if bash -l "$HERE/barkla_rtool.sh" doctor; then
  echo "Setup finished. A failed or warned check about the weights clears once the download job ends."
else
  echo "Doctor reported a failure, shown above. A check about the weights clears once the download job ends."
fi
