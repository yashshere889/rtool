"""Run an rtool experiment plan with the coding-agent tool system (codeagent) instead of rtool's Coder.

rtool supplies the inputs (the approved experiment_plan and the fetch stage's resource_manifest);
codeagent's Agent loop, tools and permission engine write and run the experiment in a workspace.
The output contract is rtool's: results.json must validate against the plan's results_schema.

    python scripts/codeagent_experiment.py RUN_DIR WORKSPACE --model M [--base-url URL] [--max-turns N]

Writes WORKSPACE/transcript.jsonl (every model turn and tool call) and WORKSPACE/driver_report.json.
No person is present in a batch job, so anything the permission engine would ask about is denied
and the model is told why; the counts show how often that happened.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import jsonschema

from codeagent import (
    Agent,
    ApprovalRequest,
    ApprovalResponse,
    OpenAICompatibleModel,
    PermissionEngine,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    default_tools,
)
from codeagent.hooks import Hooks

ALLOW = ["bash(python *)", "bash(python3 *)", "bash(mkdir *)"]
DENY = ["bash(curl *)", "bash(wget *)", "bash(pip *)", "bash(rm -rf *)", "bash(git push*)", "bash(ssh *)"]

TASK = """Implement and run the experiment described in PLAN.json (written by rtool's planner).

- Data is in ./data (copied from rtool's fetch stage). There is no network access; do not download anything.
- Python packages available: numpy, pandas, scipy, statsmodels, arch, jsonschema. You cannot install more.
- Write the code to ./experiment.py and run it with `python experiment.py`. CPU only.
- One bash call is killed after 600 s (set timeout_s up to 600), so keep the full run under about 8 minutes.
- The run must write ./results.json that validates against ./results_schema.json. Check it with
  `python -c "import json, jsonschema; jsonschema.validate(json.load(open('results.json')), json.load(open('results_schema.json'))); print('valid')"`
- Follow the plan's conditions, metrics (the primary one decides the hypothesis), statistical tests and seeds.
  If you must deviate (for example to fit the time limit), say so in results.json if the schema allows notes,
  and in your final reply.
- Run commands from the workspace; do not cd elsewhere.
When results.json is valid, reply with a short summary of the numbers and any deviations, and call no tool."""


def latest(run_dir: Path, kind: str) -> dict | None:
    versions = sorted((run_dir / "artifacts" / kind).glob("v*.json"))
    return json.loads(versions[-1].read_text()) if versions else None


def stage_workspace(run_dir: Path, ws: Path) -> tuple[dict, list[str]]:
    plan = latest(run_dir, "experiment_plan")
    if plan is None:
        raise SystemExit(f"{run_dir} has no experiment_plan artifact")
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "PLAN.json").write_text(json.dumps(plan, indent=2))
    (ws / "results_schema.json").write_text(json.dumps(plan["results_schema"], indent=2))
    copied = []
    manifest = latest(run_dir, "resource_manifest") or {"resources": []}
    for res in manifest["resources"]:
        if res.get("status") != "ready" or not res.get("path"):
            continue
        src, dst = Path(res["path"]), ws / "data" / Path(res["path"]).name
        dst.parent.mkdir(exist_ok=True)
        # Copied, not linked: codeagent asks before reading outside the workspace, and links resolve outside.
        shutil.copytree(src, dst, dirs_exist_ok=True) if src.is_dir() else shutil.copy2(src, dst)
        copied.append(f"{res['name']} -> data/{dst.name}")
    return plan, copied


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("workspace", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=8192)
    args = ap.parse_args()

    plan, copied = stage_workspace(args.run_dir, args.workspace)
    log = (args.workspace / "transcript.jsonl").open("w")
    t0 = time.time()
    calls, errors, denials, usage = Counter(), Counter(), [], Counter()

    def on_event(event: dict) -> None:
        log.write(json.dumps({"t": round(time.time() - t0, 1), **event}) + "\n")
        log.flush()
        if event["type"] == "tool_start":
            calls[event["name"]] += 1
        elif event["type"] == "tool_end" and event.get("is_error"):
            errors[event["name"]] += 1

    def approver(req: ApprovalRequest) -> ApprovalResponse:
        denials.append(req.summary)
        return ApprovalResponse(False, feedback=(
            "no person is available in this batch job; allowed: python/python3, mkdir "
            "and file tools inside the workspace"))

    model = OpenAICompatibleModel(args.model, args.base_url, max_tokens=args.max_tokens, temperature=0.0)
    complete = model.complete

    async def counted(messages, tools):  # sum token usage across turns
        turn = await complete(messages, tools)
        usage.update({k: v for k, v in turn.usage.items() if isinstance(v, int)})
        usage["turns"] += 1
        return turn

    model.complete = counted
    ctx = ToolContext(workspace=args.workspace)
    registry = ToolRegistry(default_tools())
    permissions = PermissionEngine.from_strings("accept_edits", ALLOW, DENY, approver)
    executor = ToolExecutor(registry, permissions, Hooks(), on_event=on_event)
    agent = Agent(model, registry, executor, ctx, max_turns=args.max_turns, on_event=on_event)

    final = await agent.send(TASK)

    results_path, valid, problem = args.workspace / "results.json", False, ""
    try:
        jsonschema.validate(json.loads(results_path.read_text()), plan["results_schema"])
        valid = True
    except FileNotFoundError:
        problem = "results.json was not written"
    except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
        problem = str(exc).splitlines()[0]
    report = {
        "model": args.model,
        "hypothesis_id": plan["hypothesis_id"],
        "data_copied": copied,
        "wall_s": round(time.time() - t0, 1),
        "model_turns": usage.pop("turns", 0),
        "tokens": dict(usage),
        "tool_calls": dict(calls),
        "tool_errors": dict(errors),
        "permission_denials": denials,
        "results_valid": valid,
        "results_problem": problem,
        "final_reply": final,
    }
    (args.workspace / "driver_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "final_reply"}, indent=2))
    print("\nfinal reply:\n" + final)


if __name__ == "__main__":
    asyncio.run(main())
