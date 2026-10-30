#!/bin/bash
# Stands in for scripts/rtool_ui_session.sh: serves the tunnel's Unix socket and prints the address.
S=/mnt/scratch/users/root/rtool/ui/tunnel.sock
rm -f "$S"
echo "port argument: $1"
echo "  http://127.0.0.1:$1/#token=standin"
exec python3 -c '
import http.server, socketserver, sys
class H(http.server.BaseHTTPRequestHandler):
    def address_string(self):
        return "unix"
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"tunnel-ok")
socketserver.ThreadingUnixStreamServer(sys.argv[1], H).serve_forever()
' "$S"
