#!/usr/bin/env bash
# Which way can a browser on this Mac reach a web server running inside a Barkla job?
#
#     scripts/probe_ui_access.sh [ssh-host]
#
# Run on the Mac (University VPN on). It submits ONE tiny CPU job (1 core, 1 GB, at most
# 15 minutes) that serves a test page from scratch, tries the routes, then cancels the job:
#
#   A. ssh into the job's node (jumping through the login node) and forward to a Unix socket
#      in a folder only you can open. Then nobody else on the cluster can reach the server,
#      token or not (best). Tried with the socket in scratch and in the node's /tmp.
#   B. TCP forwarding from the login node to the node's port. Works without ssh access to
#      nodes, but the server listens on the cluster network (token-protected).
#
# The node's host key is read through the login node (ssh-keyscan inside the cluster) into a
# known-hosts file of the probe's own; your ~/.ssh/known_hosts is not changed.
# If neither works, `rtool ui --mode login` (a session-bound server on the login node) remains.
# Nothing else on Barkla is touched; the probe's files live in scratch/rtool/ui-probe.
set -euo pipefail
HOST="${1:-barkla}"
LOCAL_PORT="${LOCAL_PORT:-18765}"

remote_user="$(ssh "$HOST" 'echo "$USER"')"
probe="/mnt/scratch/users/${remote_user}/rtool/ui-probe"
echo "probing $HOST as $remote_user (files in $probe)"

ssh "$HOST" "mkdir -p '$probe' && chmod 700 '$probe' && rm -f '$probe/endpoint' && cat > '$probe/probe.sh'" <<JOB
#!/bin/bash -l
#SBATCH -J rtool-ui-probe
#SBATCH -p nodes
#SBATCH -n 1
#SBATCH --mem=1G
#SBATCH -t 00:15:00
#SBATCH -o $probe/job-%j.out
cd "$probe"
tmpdir="/tmp/users/\$USER/rtool-ui-probe-\$SLURM_JOB_ID"
mkdir -p "\$tmpdir" && chmod 700 "\$tmpdir"
rm -f "$probe/ui.sock"
exec python3 - "$probe/ui.sock" "\$tmpdir/ui.sock" <<'PY'
import http.server, os, socket, socketserver, sys, threading

class Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"rtool-ui-probe-ok\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def address_string(self):
        return "local"
    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.server.server_address, fmt % args))

class Tcp(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True

class Unix(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

servers = [Tcp(("0.0.0.0", 0), Page)]
for path in sys.argv[1:]:
    try:
        servers.append(Unix(path, Page))
        os.chmod(path, 0o600)
    except OSError as exc:
        sys.stderr.write("no socket at %s: %s\n" % (path, exc))
for s in servers[1:]:
    threading.Thread(target=s.serve_forever, daemon=True).start()
with open("endpoint.tmp", "w") as f:
    f.write("%s %d %s\n" % (socket.gethostname().split(".")[0], servers[0].server_address[1], " ".join(sys.argv[1:])))
os.replace("endpoint.tmp", "endpoint")
servers[0].serve_forever()
PY
JOB

job_id="$(ssh "$HOST" "cd '$probe' && sbatch --parsable probe.sh")"
job_id="${job_id%%;*}"
echo "submitted probe job $job_id; waiting for it to start (up to 15 minutes)..."
work="$(mktemp -d)"
tunnel=""
forward=""
cleanup() {
  if [ -n "$tunnel" ]; then kill "$tunnel" 2>/dev/null || true; fi
  if [ -n "$forward" ]; then ssh -O cancel -L "$forward" "$HOST" >/dev/null 2>&1 || true; fi
  ssh "$HOST" "scancel $job_id; rm -f '$probe/ui.sock'" >/dev/null 2>&1 || true
  rm -rf "$work"
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
read -r node port sock_scratch sock_tmp <<<"$endpoint"
echo "job runs on $node, serving on port $port and on Unix sockets"

served() {  # curl the forwarded local port for a few seconds
  for _ in $(seq 15); do
    if curl -s --max-time 3 "http://127.0.0.1:$LOCAL_PORT/" 2>/dev/null | grep -q rtool-ui-probe-ok; then return 0; fi
    sleep 1
  done
  return 1
}

# Route A: the node's host key as the cluster sees it, kept in the probe's own file.
ssh "$HOST" "ssh-keyscan -t ed25519 $node 2>/dev/null" > "$work/known_hosts" || true
node_ssh=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ExitOnForwardFailure=yes
  -o UserKnownHostsFile="$work/known_hosts" -o StrictHostKeyChecking=yes -J "$HOST" "$remote_user@$node")
route_a_ssh=no
if [ -s "$work/known_hosts" ] && "${node_ssh[@]}" true 2>"$work/ssh.err"; then route_a_ssh=yes; fi

try_socket() {  # sets $result
  "${node_ssh[@]}" -N -L "127.0.0.1:$LOCAL_PORT:$1" >/dev/null 2>>"$work/ssh.err" &
  tunnel=$!
  result=no
  if served; then result=yes; fi
  kill "$tunnel" 2>/dev/null || true
  wait "$tunnel" 2>/dev/null || true
  tunnel=""
}
route_a_scratch=no
route_a_tmp=no
if [ "$route_a_ssh" = yes ]; then
  try_socket "$sock_scratch"; route_a_scratch="$result"
  try_socket "$sock_tmp"; route_a_tmp="$result"
fi

route_b=no
forward="$LOCAL_PORT:$node:$port"
if ssh -o ExitOnForwardFailure=yes -O forward -L "$forward" "$HOST" >/dev/null 2>&1; then
  if served; then route_b=yes; fi
  ssh -O cancel -L "$forward" "$HOST" >/dev/null 2>&1 || true
else  # no ssh control master: a tunnel of its own
  ssh -o ExitOnForwardFailure=yes -N -L "$forward" "$HOST" >/dev/null 2>&1 &
  tunnel=$!
  if served; then route_b=yes; fi
  kill "$tunnel" 2>/dev/null || true
  tunnel=""
fi
forward=""

echo
echo "A. ssh to the job's node:                         $route_a_ssh"
echo "   Unix socket in scratch, forwarded:             $route_a_scratch"
echo "   Unix socket in the node's /tmp, forwarded:     $route_a_tmp"
echo "B. TCP forwarding through the login node:         $route_b"
if [ "$route_a_ssh" = no ] && [ -s "$work/ssh.err" ]; then
  echo "   (ssh to the node said: $(tail -1 "$work/ssh.err"))"
fi
if [ "$route_a_scratch" = yes ] || [ "$route_a_tmp" = yes ]; then
  echo "=> rtool ui --mode job (the default): a Unix socket only you can open; scripts/rtool-ui tunnels to it"
elif [ "$route_b" = yes ]; then
  echo "=> rtool ui --mode job --bind cluster (token-protected; scripts/rtool-ui --bind cluster tunnels via $HOST)"
else
  echo "=> rtool ui --mode login (a server that lives only while your ssh session does)"
fi
