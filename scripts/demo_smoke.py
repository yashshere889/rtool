#!/usr/bin/env python3
"""Check a running `rtool ui --mode local --demo`: pages, script types, and a new run end to end.

    python scripts/demo_smoke.py --port 8765 --token-file token.txt

Standard library only, so it runs anywhere, including a Windows test machine.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PAGES = ("literature", "hypotheses", "plan", "code", "results", "report", "review")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--wait", type=float, default=180, help="seconds to wait for the server and for the new run")
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    problems: list[str] = []

    def request(path: str, body: dict | None = None) -> tuple[int, str, str]:
        token = args.token_file.read_text(encoding="utf-8").strip()
        req = urllib.request.Request(
            base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "X-Requested-With": "rtool",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.status, response.headers.get("content-type", ""), response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            return error.code, "", error.read().decode("utf-8", "replace")

    def fetch_status(path: str) -> int:
        token = args.token_file.read_text(encoding="utf-8").strip()
        req = urllib.request.Request(base + path, headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                response.read()
                return response.status
        except urllib.error.HTTPError as error:
            return error.code

    deadline = time.monotonic() + args.wait
    while True:
        try:
            status, _, shell = request("/")
            break
        except (OSError, urllib.error.URLError):
            if time.monotonic() > deadline:
                print("FAILED: the server never answered")
                return 1
            time.sleep(2)
    if status != 200:
        problems.append(f"/ answered {status}")

    for asset in sorted(set(re.findall(r'(?:src|href)="/?(assets/[^"]+\.(?:js|css))"', shell))):
        status, kind, _ = request("/" + asset)
        wanted = "javascript" if asset.endswith(".js") else "css"
        print(f"{asset}: {status} {kind}")
        if status != 200 or wanted not in kind:
            problems.append(f"{asset} answered {status} as {kind!r}")

    status, _, text = request("/api/runs")
    runs = json.loads(text) if status == 200 else []
    print(f"demo runs: {sorted(r['run_id'] for r in runs)}")
    if len(runs) < 6:
        problems.append(f"expected 6 demo runs, found {len(runs)}")

    for page in PAGES:
        status, _, text = request(f"/api/runs/sleep-demo/6-done/{page}")
        print(f"{page}: {status}")
        if status != 200:
            problems.append(f"{page} answered {status}: {text[:200]}")
        elif "\\\\work" in text or "work\\\\" in text:
            problems.append(f"{page} hands out a Windows path")

    question = {"topic": "Smoke test: does sound in sleep help memory?", "review": "automatic"}
    status, _, text = request("/api/runs", question)
    if status != 201:
        problems.append(f"new run answered {status}: {text[:200]}")
    else:
        run = json.loads(text)["run"]
        path = f"/api/runs/{run['project']}/{run['run_id']}"
        deadline = time.monotonic() + args.wait
        outcome = None
        while time.monotonic() < deadline:
            _, _, text = request(path)
            detail = json.loads(text)
            outcome = detail.get("last_outcome")
            if outcome == "done" or (detail.get("worker") or {}).get("last_error"):
                break
            time.sleep(2)
        print(f"new run: {outcome}")
        if outcome != "done":
            stages = [(s["id"], s["status"]) for s in detail.get("stages", [])]
            problems.append(f"new run ended as {outcome}: {detail.get('worker')} {stages}")

    # Files the pages open or download.
    _, _, text = request("/api/runs/sleep-demo/6-done/report")
    figures = json.loads(text).get("figures") or []
    for path in [f"/api/runs/sleep-demo/6-done/files/{f}" for f in figures[:1]] + [
        "/api/runs/sleep-demo/6-done/report/bundle",
        "/api/runs/sleep-demo/6-done/code/archive",
        "/api/runs/sleep-demo/6-done/literature/search?q=sleep",
    ]:
        status = fetch_status(path)
        print(f"{path.rsplit('/', 1)[-1]}: {status}")
        if status != 200:
            problems.append(f"{path} answered {status}")

    # The decisions a researcher makes, one on each run stopped at a checkpoint.
    for run_id in ("1-literature", "2-hypotheses", "3-plan", "4-gpu-runs", "5-final-report"):
        path = f"/api/runs/sleep-demo/{run_id}"
        _, _, text = request(path)
        gate = json.loads(text)["open_gates"][0]
        kind = "select" if gate["kind"] == "select" else "approve"
        status, _, text = request(
            path + "/actions",
            {
                "kind": kind,
                "stage": gate["stage"],
                "gate": gate["gate"],
                "payload": {"index": 0} if kind == "select" else {},
            },
        )
        if status >= 300:
            problems.append(f"{run_id}: {kind} answered {status}: {text[:200]}")
            continue
        deadline = time.monotonic() + args.wait
        detail: dict = {}
        while time.monotonic() < deadline:
            _, _, text = request(path)
            detail = json.loads(text)
            gates = [g["gate"] for g in detail.get("open_gates", [])]
            settled = not detail.get("queued_actions") and not (detail.get("worker") or {}).get("active")
            if settled and gate["gate"] not in gates:
                break
            time.sleep(1)
        print(f"{run_id}: {kind} -> {detail.get('last_outcome')} {[g['gate'] for g in detail.get('open_gates', [])]}")
        failed = [s["id"] for s in detail.get("stages", []) if s["status"] == "failed"]
        error = (detail.get("worker") or {}).get("last_error")
        if failed or error or gate["gate"] in [g["gate"] for g in detail.get("open_gates", [])]:
            problems.append(f"{run_id}: after {kind}: failed stages {failed}, error {error}")

    for problem in problems:
        print("FAILED:", problem)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
