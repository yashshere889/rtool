/** The experiment plan in everyday words: where resources live, download states, and what an edit changes. */
import type { DownloadItem, ExperimentPlan, ResourceSpec } from "@/api/types";

import { sameJson } from "./equal";
import type { Tone } from "./stages";

export const RESOURCE_KINDS: [ResourceSpec["kind"], string][] = [
  ["dataset", "Dataset"],
  ["model", "Model"],
  ["code", "Software"],
  ["other", "Other"],
];

export const RESOURCE_SOURCES: [ResourceSpec["source"], string][] = [
  ["huggingface", "Hugging Face"],
  ["url", "Web address"],
  ["torchvision", "torchvision"],
  ["pip", "Python package (pip)"],
  ["manual", "By hand"],
];

/** A page where the researcher can look the resource up. */
export function resourceLink(r: Pick<ResourceSpec, "source" | "identifier" | "kind">): string | null {
  const id = r.identifier.trim();
  if (!id) return null;
  switch (r.source) {
    case "huggingface":
      return r.kind === "dataset" ? `https://huggingface.co/datasets/${id}` : `https://huggingface.co/${id}`;
    case "url":
      return /^https?:\/\//.test(id) ? id : null;
    case "pip":
      return `https://pypi.org/project/${id.split(/[=<>![ ;]/)[0]}/`;
    case "torchvision":
      return "https://pytorch.org/vision/stable/datasets.html";
    default:
      return null;
  }
}

/** Whether the planner (or the download step) could confirm the resource exists. */
export function resourceCheck(r: ResourceSpec): { label: string; tone: Tone } {
  if (r.verified) return { label: "Found", tone: "ok" };
  if (r.note.startsWith("changed by hand")) return { label: "Not checked yet", tone: "warn" };
  return { label: "Not found", tone: "bad" };
}

export const DOWNLOAD_STATES: Record<DownloadItem["status"], { label: string; tone: Tone }> = {
  waiting: { label: "Waiting", tone: "neutral" },
  ready: { label: "Downloaded", tone: "ok" },
  deferred: { label: "Installed when the code runs", tone: "info" },
  manual: { label: "Needs you: get it by hand", tone: "warn" },
  failed: { label: "Failed", tone: "bad" },
};

export interface Change {
  label: string;
  detail: string;
}

const TEXT: [keyof ExperimentPlan, string][] = [
  ["objective", "Objective"],
  ["design", "Design"],
  ["expected_results", "Expected results"],
];

const LISTS: [keyof ExperimentPlan, string, (item: never) => string][] = [
  ["independent_variables", "Independent variables", (v: { name: string }) => v.name],
  ["dependent_variables", "Dependent variables", (v: { name: string }) => v.name],
  ["controls", "Controls", (v: string) => v],
  ["conditions", "Conditions", (v: { name: string }) => v.name],
  ["ablations", "Ablations", (v: { name: string }) => v.name],
  ["metrics", "Metrics", (v: { name: string }) => v.name],
  ["statistical_tests", "Statistical tests", (v: { test: string }) => v.test],
  ["procedure", "Procedure", (v: string) => v],
  ["resources", "Resources", (v: { name: string }) => v.name],
  ["risks", "Risks", (v: string) => v],
];

const COMPUTE: [keyof ExperimentPlan["compute"], string][] = [
  ["partition", "partition"],
  ["gpus_per_run", "GPUs per run"],
  ["gpu_memory_gb", "GPU memory (GB)"],
  ["hours_per_run", "hours per run"],
  ["time_limit", "time limit"],
  ["justification", "reasoning"],
];

const same = sameJson;

function listChange(before: unknown[], after: unknown[], name: (item: never) => string): string {
  const parts: string[] = [];
  const shared = Math.min(before.length, after.length);
  const changed = Array.from({ length: shared }, (_, i) => i).filter((i) => !same(before[i], after[i]));
  if (after.length > before.length) parts.push(`${after.length - before.length} added`);
  if (before.length > after.length) parts.push(`${before.length - after.length} removed`);
  if (changed.length) {
    const names = changed.map((i) => name(after[i] as never)).filter(Boolean).slice(0, 3);
    parts.push(`${changed.length} changed${names.length ? ` (${names.join(", ")})` : ""}`);
  }
  return parts.join(", ");
}

/** What the researcher changed, field by field, in words. */
export function planChanges(before: ExperimentPlan, after: ExperimentPlan): Change[] {
  const changes: Change[] = [];
  for (const [key, label] of TEXT) {
    if (!same(before[key], after[key])) changes.push({ label, detail: "rewritten" });
  }
  for (const [key, label, name] of LISTS) {
    const detail = listChange(before[key] as unknown[], after[key] as unknown[], name);
    if (detail) changes.push({ label, detail });
  }
  if (before.seeds.length !== after.seeds.length) {
    changes.push({ label: "Seeds per condition", detail: `${before.seeds.length} → ${after.seeds.length}` });
  }
  for (const [key, label] of COMPUTE) {
    if (!same(before.compute[key], after.compute[key])) {
      const detail = key === "justification" ? "rewritten" : `${before.compute[key] ?? "–"} → ${after.compute[key] ?? "–"}`;
      changes.push({ label: `Compute: ${label}`, detail });
    }
  }
  return changes;
}

/** Seeds 0..n-1 for "n seeds per condition". */
export function seedList(n: number): number[] {
  return Array.from({ length: Math.max(1, Math.min(50, Math.floor(n) || 1)) }, (_, i) => i);
}

/** The pre-registered primary outcome: the marked metric, or the first one in plans written before metrics were marked. */
export function primaryMetric(plan: Pick<ExperimentPlan, "metrics">): ExperimentPlan["metrics"][number] | undefined {
  return plan.metrics.find((m) => m.primary) ?? plan.metrics[0];
}
