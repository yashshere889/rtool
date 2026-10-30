#!/bin/bash -l
# The Barkla end of one login: start or find the UI job and connect it to the login's tunnel, so
# a computer that cannot reuse an ssh login (Windows) needs one password, not one per step:
#
#     ssh -t -L 127.0.0.1:8780:/mnt/scratch/users/$USER/rtool/ui/tunnel.sock $USER@barkla \
#       "bash -l ~/'Research tool'/scripts/rtool_ui_session.sh 8780"
#
# Runs on the login node for as long as the ssh session lasts; it only forwards bytes. The job
# keeps running afterwards until its hours are used up or `rtool ui --stop` ends it.
set -euo pipefail
PORT="${1:-8780}"
HOURS="${RTOOL_UI_HOURS:-8}"
STATE_DIR="${RTOOL_UI_STATE_DIR:-/mnt/scratch/users/$USER/rtool/ui}"
RTOOL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/barkla_rtool.sh"
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "the port must be a number" >&2; exit 2; }

field() { # field NAME < job JSON
  python3 -I -c 'import json, sys; v = (json.load(sys.stdin) or {}).get(sys.argv[1]); print("" if v is None else v)' "$1"
}

echo "Starting or finding the UI job..."
job="$(bash -l "$RTOOL" ui --json --wait 60 --hours "$HOURS" --bind socket --state-dir "$STATE_DIR")"
said=""
while [ "$(field serving <<<"$job")" != True ]; do
  if [ -z "$(field job_id <<<"$job")" ]; then
    echo "The UI job is gone; its log is in $STATE_DIR" >&2
    exit 1
  fi
  line="UI job $(field job_id <<<"$job"): $(field state <<<"$job" | tr '[:upper:]' '[:lower:]') $(field reason <<<"$job")"
  [ "$line" != "$said" ] && echo "$line (waiting)"
  said="$line"
  sleep 15
  job="$(bash -l "$RTOOL" ui --status --json --state-dir "$STATE_DIR")"
done

job_id="$(field job_id <<<"$job")"
node="$(field node <<<"$job")"
dir="$(field directory <<<"$job")"
sock="$(field socket <<<"$job")"
tunnel_sock="$STATE_DIR/tunnel.sock"
ssh -N -o BatchMode=yes -o LogLevel=ERROR -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -o UserKnownHostsFile="$STATE_DIR/known_hosts" -o StrictHostKeyChecking=accept-new \
  -o StreamLocalBindUnlink=yes -o StreamLocalBindMask=0177 \
  -L "$tunnel_sock:$sock" "$node" &
tunnel=$!
trap 'kill "$tunnel" 2>/dev/null || true; rm -f "$tunnel_sock"' EXIT
trap 'exit 0' HUP INT TERM

answers() { curl -sf --max-time 10 --unix-socket "$tunnel_sock" "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; }
for _ in $(seq 30); do
  answers && break
  kill -0 "$tunnel" 2>/dev/null || { echo "Could not connect to $node" >&2; exit 1; }
  sleep 1
done
answers || { echo "The UI did not answer; its log is in $STATE_DIR" >&2; exit 1; }

echo
echo "Research Tool is ready (job $job_id on $node). Open this address in your browser:"
echo
echo "  http://127.0.0.1:$PORT/#token=$(cat "$dir/token")"
echo
echo "Keep this window open while you use it. Closing it does not stop your runs."
misses=0
while kill -0 "$tunnel" 2>/dev/null; do
  sleep 30 & # in the background so a hangup ends this at once, not after the sleep
  wait $! || true
  if answers; then misses=0; else misses=$((misses + 1)); fi
  if [ "$misses" -ge 2 ]; then
    echo "The UI stopped answering (the job ended, or the connection dropped). Start this again to reconnect." >&2
    exit 1
  fi
done
echo "The connection closed. Start this again to reconnect." >&2
exit 1
