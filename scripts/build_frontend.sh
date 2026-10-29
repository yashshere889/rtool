#!/usr/bin/env bash
# Build the web app into src/research_tool/web/static, where `rtool ui` serves it from.
# Commit the result: the cluster runs the UI without Node.
#
#     scripts/build_frontend.sh            # needs Node >= 22 and the dev venv (.venv) with the web extra
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" -m research_tool.web.openapi frontend/src/api/openapi.json
cd frontend
[ -d node_modules ] || npm ci --no-audit --no-fund
npm run api-types
npm run build
