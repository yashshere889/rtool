#!/usr/bin/env bash
# Which way can a browser on this Mac reach a web server running inside a Barkla job?
#
#     scripts/probe_ui_access.sh [ssh-host]
#
# Run on the Mac (University VPN on). It submits ONE tiny CPU job (1 core, 1 GB, at most
# 15 minutes) that serves a test page from scratch, tries two routes, then cancels the job:
#
#   A. ssh into the job's node, jumping through the login node. If this works, `rtool ui`
#      can bind to 127.0.0.1 on the node, where no other user can reach it (best).
#   B. TCP forwarding from the login node to the node's port. Works without ssh access to
#      nodes, but the server must listen on the cluster network (token-protected).
#
# If neither works, `rtool ui --mode login` (a session-bound server on the login node) remains.
# Nothing else on Barkla is touched; the probe's files live in scratch/rtool/ui-probe.
set -euo pipefail
HOST="${1:-barkla}"
LOCAL_PORT="${LOCAL_PORT:-18765}"

remote_user="$(ssh "$HOST" 'echo "$USER"')"
probe="/mnt/scratch/users/${remote_user}/rtool/ui-probe"
echo "probing $HOST as $remote_user (files in $probe)"

ssh "$HOST" "mkdir -p '$probe' && rm -f '$probe/endpoint' && cat > '$probe/probe.sh'" <<JOB
#!/bin/bash -l
#SBATCH -J rtool-ui-probe
#SBATCH -p nodes
#SBATCH -n 1
#SBATCH --mem=1G
#SBATCH -t 00:15:00
#SBATCH -o $probe/job-%j.out
cd "$probe"
port=\$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
echo rtool-ui-probe-ok > index.html
echo "\$(hostname -s) \$port" > endpoint.tmp && mv endpoint.tmp endpoint
exec python3 -m http.server "\$port" --bind 0.0.0.0
JOB

job_id="$(ssh "$HOST" "cd '$probe' && sbatch --parsable probe.sh")"
job_id="${job_id%%;*}"
echo "submitted probe job $job_id; waiting for it to start (up to 15 minutes)..."
forward=""
cleanup() {
  if [ -n "$forward" ]; then ssh -O cancel -L "$forward" "$HOST" >/dev/null 2>&1 || true; fi
  ssh "$HOST" "scancel $job_id" >/dev/null 2>&1 || true
}
trap cleanup EXIT

endpoint=""
for _ in $(seq 180); do
  endpoint="$(ssh "$HOST" "cat '$probe/endpoint' 2>/dev/null" || true)"
  [ -n "$endpoint" ] && break
  sleep 5
done
if [ -z "$endpoint" ]; then
  echo "the probe job did not start within 15 minutes; try again later (queue: ssh $HOST squeue -u $remote_user)"
  exit 1
fi
read -r node port <<<"$endpoint"
echo "job runs on $node, serving on port $port"

route_a=no
if ssh -o BatchMode=yes -o ConnectTimeout=20 -J "$HOST" "$remote_user@$node" \
  "curl -s --max-time 10 http://127.0.0.1:$port/" 2>/dev/null | grep -q rtool-ui-probe-ok; then
  route_a=yes
fi

route_b=no
forward="$LOCAL_PORT:$node:$port"
if ssh -o ExitOnForwardFailure=yes -O forward -L "$forward" "$HOST" >/dev/null 2>&1 \
  || ssh -o ExitOnForwardFailure=yes -f -N -L "$forward" "$HOST"; then
  if curl -s --max-time 10 "http://127.0.0.1:$LOCAL_PORT/" | grep -q rtool-ui-probe-ok; then
    route_b=yes
  fi
fi

echo
echo "A. ssh to the job's node (loopback-only server): $route_a"
echo "B. forwarding through the login node:            $route_b"
if [ "$route_a" = yes ]; then
  echo "=> use: rtool ui --mode job (binds 127.0.0.1 on the node; the launcher tunnels with -J $HOST)"
elif [ "$route_b" = yes ]; then
  echo "=> use: rtool ui --mode job --bind cluster (token-protected; the launcher tunnels via $HOST)"
else
  echo "=> use: rtool ui --mode login (a server that lives only while your ssh session does)"
fi
