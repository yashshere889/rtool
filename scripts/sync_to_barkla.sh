#!/usr/bin/env bash
# Copy the source tree to Barkla, into ~/"Research tool" (code is small; $HOME is backed up).
#
#     scripts/sync_to_barkla.sh [ssh-host] [dest-dir-under-home]
#
# The folder on Barkla mirrors this one: files deleted here are deleted there too, so keep
# notes and data elsewhere (runs live in scratch). Excludes virtualenvs, caches, local runs,
# git metadata and the web frontend's sources (Barkla gets the built bundle in src/).
set -euo pipefail
HOST="${1:-barkla}"
DEST="${2:-Research tool}"
cd "$(dirname "$0")/.."
# The remote shell splits the destination on spaces, so quote it for that shell. RSYNC_OLD_ARGS
# makes rsync 3.2.4+ pass it through as-is, like the rsync 2.6.9 that ships with macOS.
export RSYNC_OLD_ARGS=1
rsync -az --delete \
  --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' --exclude '*.egg-info/' \
  --exclude '.pytest_cache/' --exclude '.ruff_cache/' --exclude '/runs/' --exclude '/rtool-runs/' \
  --exclude '/frontend/' --exclude 'node_modules/' --exclude '.DS_Store' \
  ./ "$HOST:$(printf '%q' "$DEST")/"
echo "synced to $HOST:~/$DEST"
