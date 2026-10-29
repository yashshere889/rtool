/** The coding stage in everyday words: its steps, what a repair was, and the state of each run. */
import type { CodeView, RunCell } from "@/api/types";

import type { Tone } from "./stages";

export const STEPS = [
  { id: "write", label: "Writing the code" },
  { id: "smoke", label: "Testing on a CPU" },
  { id: "approve", label: "Your OK for GPU time" },
  { id: "run", label: "Running on the cluster" },
  { id: "finish", label: "Analysing the results" },
] as const;

const STEP_OF: Record<string, number> = { write: 0, smoke: 1, approve: 2, submit: 3, collect: 3, finish: 4 };

export type StepState = "done" | "current" | "waiting" | "failed" | "attention";

/** Where each step stands, from the stage status and the coding agent's last announced phase. */
export function stepStates(view: Pick<CodeView, "status" | "phase">): StepState[] {
  if (view.status === "done") return STEPS.map(() => "done");
  const current = view.phase ? (STEP_OF[view.phase] ?? 0) : -1;
  return STEPS.map((_, i) => {
    if (i < current) return "done";
    if (i > current) return "waiting";
    if (view.status === "failed") return "failed";
    if (view.status === "awaiting_gate") return "attention";
    return view.status === "pending" || view.status === "stale" ? "waiting" : "current";
  });
}

export const REPAIRS: Record<string, string> = {
  static_check: "A check before running found a problem",
  syntax_error: "The code had a syntax error",
  traceback: "The code crashed",
  missing_module: "A Python package was missing",
  install_failed: "Packages could not be installed",
  file_not_found: "A data file was not where the code looked",
  not_implemented: "Part of the code was left unwritten",
  contract: "The output did not match what the plan asks for",
  smoke_timeout: "The CPU test took too long",
  cpu_oom: "It ran out of memory",
  cuda_oom: "It ran out of GPU memory",
  time_limit: "A run hit its time limit",
  not_run: "Runs did not start (or were stopped by the cluster)",
  cancelled: "The jobs were cancelled",
  human_edit: "You edited the code",
  failed: "A run failed",
};

export function repairLabel(signature: string): string {
  return REPAIRS[signature] ?? signature.replace(/_/g, " ");
}

export const RUN_STATES: Record<RunCell["status"], { label: string; tone: Tone }> = {
  waiting: { label: "Waiting", tone: "neutral" },
  running: { label: "Queued or running", tone: "info" },
  ok: { label: "Finished", tone: "ok" },
  failed: { label: "Failed", tone: "bad" },
  missing: { label: "No result", tone: "warn" },
};

/** Four significant digits: enough to compare, without noise. */
export function num(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "–";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 0.001 || abs >= 1e6)) return value.toExponential(2);
  return Number(value.toPrecision(4)).toString();
}

export function pValue(p: number | null | undefined): string {
  if (p === null || p === undefined) return "–";
  return p < 0.001 ? "< 0.001" : p.toFixed(3);
}

/** The draft name the server uses for an edited file. */
export function fileDraftKey(path: string): string {
  return `file_${path.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "")}`.slice(0, 63);
}
