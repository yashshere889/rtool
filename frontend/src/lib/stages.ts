/** Stages and checkpoints in everyday words: what each is, and what its state means for the researcher. */
import type { OpenGate, RunDetail, RunSummary, StageSummary } from "@/api/types";

import { humanize } from "./format";

export type Tone = "neutral" | "info" | "warn" | "ok" | "bad" | "stale";

interface StageInfo {
  title: string;
  description: string;
}

export const STAGES: Record<string, StageInfo> = {
  literature: {
    title: "Literature",
    description:
      "Searches the literature, screens what it finds, reads the open-access papers and maps their findings, " +
      "methods and gaps. Every extracted claim is checked against a quote from the paper.",
  },
  hypotheses: {
    title: "Hypotheses",
    description:
      "Drafts testable hypotheses from the gaps in the literature, checks each against the papers it cites, " +
      "judges novelty and feasibility, and offers you the best three.",
  },
  plan: {
    title: "Experiment plan",
    description:
      "Designs the experiment for your hypothesis: conditions, metrics, statistics, the datasets and models it " +
      "needs (checked online) and the compute it will use.",
  },
  fetch: {
    title: "Datasets & models",
    description: "Downloads the datasets and models the approved plan lists, within your storage quota.",
  },
  code: {
    title: "Code & runs",
    description:
      "Writes the experiment code, tests it on a CPU, runs it on the GPUs you approve, repairs failures and " +
      "computes every statistic and figure from the results.",
  },
  report: {
    title: "Report",
    description:
      "Writes the report. Numbers come straight from the results and citations only from papers that were read.",
  },
  review: {
    title: "Review",
    description:
      "Checks the report, has it reviewed three times independently, edits the text, and sends problems with " +
      "the experiment back to the stage that can fix them.",
  },
};

export function stageTitle(id: string): string {
  return STAGES[id]?.title ?? humanize(id);
}

interface GateInfo {
  title: string;
  approve: string;
  reject: string;
  help: string;
}

const GATES: Record<string, GateInfo> = {
  literature_review: {
    title: "Review the literature",
    approve: "Approve & continue",
    reject: "Request changes",
    help: "Approving moves on to the hypotheses. Requesting changes searches and screens again with your comments.",
  },
  hypothesis_selection: {
    title: "Choose a hypothesis",
    approve: "Choose",
    reject: "Ask for new hypotheses",
    help: "The experiment will be planned for the hypothesis you choose.",
  },
  experiment_plan: {
    title: "Approve the experiment plan",
    approve: "Approve plan & start downloads",
    reject: "Request changes",
    help: "Approving lets the tool download the datasets and models the plan lists.",
  },
  gpu_runs: {
    title: "Approve GPU time",
    approve: "Approve GPU runs",
    reject: "Request code changes",
    help: "Nothing runs on a GPU until you approve. Requesting changes has the code revised and tested again.",
  },
  final_report: {
    title: "Approve the final report",
    approve: "Approve & finish",
    reject: "Request another round",
    help: "Requesting another round sends your comments to the writer, then the report is reviewed again.",
  },
  revision_limit: {
    title: "Revision limit reached",
    approve: "Keep the current results",
    reject: "One more round",
    help: "A later stage asked for more revisions than allowed. Decide whether to accept what there is.",
  },
};

export function gateInfo(gateId: string): GateInfo {
  return (
    GATES[gateId] ?? {
      title: humanize(gateId),
      approve: "Approve",
      reject: "Request changes",
      help: "Approve to continue, or request changes with your comments.",
    }
  );
}

export interface StageState {
  label: string;
  tone: Tone;
  detail?: string;
}

/** A stage's state in plain words. ``run`` tells whether something is advancing the run right now. */
export function stageState(stage: StageSummary, run?: Pick<RunDetail, "stages" | "worker">): StageState {
  switch (stage.status) {
    case "done":
      return { label: "Done", tone: "ok" };
    case "awaiting_gate":
      return { label: "Your decision needed", tone: "warn", detail: stage.gate ? gateInfo(stage.gate.gate).title : "" };
    case "running":
      return { label: "Working", tone: "info" };
    case "suspended":
      return {
        label: "Jobs running on the cluster",
        tone: "info",
        detail: stage.waiting_on.length ? `Slurm jobs ${stage.waiting_on.join(", ")}` : undefined,
      };
    case "stale":
      return { label: "Will run again", tone: "stale", detail: stage.stale_reason ?? undefined };
    case "failed":
      return { label: "Needs attention", tone: "bad", detail: stage.last_error ?? undefined };
    case "pending": {
      const next = run?.stages.find((s) => s.status !== "done");
      if (next?.id === stage.id && run?.worker?.active) return { label: "Working", tone: "info" };
      if (stage.attempt > 0) return { label: "Paused", tone: "neutral" };
      return { label: "Not started", tone: "neutral" };
    }
    default:
      return { label: humanize(String(stage.status)), tone: "neutral" };
  }
}

export interface Headline {
  tone: Tone;
  text: string;
  stage?: string;
  gate?: OpenGate;
}

/** The one-line answer to "where is this run?", most urgent first. */
export function runHeadline(run: RunSummary | RunDetail): Headline {
  if (run.error) return { tone: "bad", text: "This run cannot be read" };
  const gate = run.open_gates[0];
  if (gate) return { tone: "warn", text: `Your decision needed: ${gateInfo(gate.gate).title}`, stage: gate.stage, gate };
  const failed = run.stages.find((s) => s.status === "failed");
  if (failed) return { tone: "bad", text: `Needs attention: ${stageTitle(failed.id)}`, stage: failed.id };
  if (run.stages.length && run.stages.every((s) => s.status === "done")) return { tone: "ok", text: "Finished" };
  const suspended = run.stages.find((s) => s.status === "suspended");
  if (suspended) return { tone: "info", text: "Experiments running on the cluster", stage: suspended.id };
  const current = run.stages.find((s) => s.status !== "done");
  const worker = "worker" in run ? run.worker : undefined;
  if (current && (worker?.active || current.status === "running"))
    return { tone: "info", text: `Working: ${stageTitle(current.id)}`, stage: current.id };
  if (current && run.last_outcome === "new") return { tone: "neutral", text: "Waiting to start", stage: current.id };
  return { tone: "neutral", text: current ? `Paused before ${stageTitle(current.id)}` : "Idle", stage: current?.id };
}

export const TONE_STYLES: Record<Tone, string> = {
  neutral: "bg-muted text-ink-soft",
  info: "bg-info-soft text-info",
  warn: "bg-warn-soft text-warn",
  ok: "bg-ok-soft text-ok",
  bad: "bg-bad-soft text-bad",
  stale: "bg-stale-soft text-stale",
};

export const TONE_DOT: Record<Tone, string> = {
  neutral: "bg-line",
  info: "bg-info",
  warn: "bg-warn",
  ok: "bg-ok",
  bad: "bg-bad",
  stale: "bg-stale",
};
