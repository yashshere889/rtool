/** Turning the run's event log into an activity feed a researcher can read. */
import type { RunEvent } from "@/api/types";

import { count, humanize } from "./format";
import { gateInfo, stageTitle, type Tone } from "./stages";

export interface Described {
  text: string;
  tone: Tone;
}

type Data = Record<string, unknown>;

const num = (value: unknown): number => (typeof value === "number" ? value : Number(value) || 0);
const str = (value: unknown): string => (value === undefined || value === null ? "" : String(value));

const DESCRIBE: Record<string, (data: Data, stage: string) => Described | null> = {
  stage_started: (d, s) => ({ text: `${s} started${d.resumed ? " (continuing)" : ""}`, tone: "info" }),
  stage_finished: (_d, s) => ({ text: `${s} finished`, tone: "ok" }),
  stage_failed: (d, s) => ({ text: `${s} failed: ${str(d.error)}`, tone: "bad" }),
  stage_retrying: (d, s) => ({ text: `${s} hit a problem and is trying again: ${str(d.error)}`, tone: "warn" }),
  stage_stale: (d, s) => ({ text: `${s} will run again: ${str(d.reason)}`, tone: "stale" }),
  stage_suspended: (d, s) => ({ text: `${s} is waiting for cluster jobs (${str(d.reason)})`, tone: "info" }),
  stage_deferred: (_d, s) => ({ text: `${s} needs the AI model: moving to a GPU`, tone: "info" }),
  stage_waiting_for_revisions: (d, s) => ({
    text: `${s} is waiting for revisions of ${(d.targets as string[] | undefined)?.map(stageTitle).join(", ")}`,
    tone: "stale",
  }),
  gate_opened: (d) => ({
    text: `${d.reopened ? "Reopened at your request" : "Your decision needed"}: ${gateInfo(str(d.gate)).title}`,
    tone: "warn",
  }),
  gate_resolved: (d) => ({
    text: `${gateInfo(str(d.gate)).title}: ${d.approved ? "approved" : "changes requested"}${d.auto ? " (automatic)" : ""}`,
    tone: d.approved ? "ok" : "stale",
  }),
  gate_auto_resolved: (d) => ({ text: `${gateInfo(str(d.gate)).title}: approved automatically`, tone: "ok" }),
  action_applied: (d) => ({ text: `Your decision (${str(d.action)}) was applied`, tone: "ok" }),
  action_rejected: (d) => ({ text: `Your decision (${str(d.action)}) was refused: ${str(d.error)}`, tone: "bad" }),
  artifact_committed: () => null, // shown through the stage that produced it
  jobs_finished: (d) => ({ text: `Cluster jobs finished (${Object.keys((d.jobs as Data) ?? {}).join(", ")})`, tone: "info" }),
  jobs_cancelled: (d) => ({ text: `Cancelled jobs ${(d.jobs as string[] | undefined)?.join(", ")}`, tone: "warn" }),
  episode_submitted: (d) => ({
    text: d.needs_llm
      ? `Queued a GPU worker (job ${str(d.job)}${d.expected_start ? `, expected to start ${str(d.expected_start).replace("T", " ")}` : ""})`
      : `Queued a worker (job ${str(d.job)})`,
    tone: "neutral",
  }),
  episode_started: () => null,
  episode_finished: () => null,
  stop_requested: (d) => ({ text: `Stopped (${count((d.jobs as unknown[] | undefined)?.length ?? 0, "job")} cancelled)`, tone: "warn" }),
  change_request_raised: (d) => ({
    text: `Asked ${stageTitle(str(d.target))} to revise: ${str(d.reason)}`,
    tone: "stale",
  }),
  revision_requested: (d, s) => ({ text: `${s} will revise its work (round ${str(d.round)})`, tone: "stale" }),
  literature_search_done: (d) => ({ text: `Search found ${count(num(d.candidates), "candidate paper")}`, tone: "info" }),
  literature_screened: (d) => ({ text: `Screening kept ${count(num(d.included), "paper")}`, tone: "info" }),
  literature_snowball: (d) => ({ text: `Following citations found ${count(num(d.new_candidates), "new candidate")}`, tone: "info" }),
  literature_downloaded: (d) => ({
    text: `Downloaded ${count(num(d.downloaded), "paper")}${num(d.unavailable) ? `; ${num(d.unavailable)} had no open-access copy` : ""}`,
    tone: "info",
  }),
  literature_manual_paper: (d) => ({ text: `Added your paper ${str(d.paper)}`, tone: "ok" }),
  literature_plan: (d) => ({ text: `Planned ${count((d.queries as unknown[] | undefined)?.length ?? 0, "search query", "search queries")}`, tone: "neutral" }),
  literature_extracted: (d) => ({
    text: `Read ${count(num(d.papers), "paper")} and extracted their findings${num(d.failures) ? ` (${num(d.failures)} could not be read)` : ""}`,
    tone: "info",
  }),
  literature_small_corpus: (d) => ({ text: `Only ${count(num(d.included), "relevant paper")} found (aimed for ${str(d.wanted)})`, tone: "warn" }),
  literature_search_failed: (d) => ({ text: `A search on ${str(d.source)} failed: ${str(d.error)}`, tone: "warn" }),
  literature_snowball_failed: () => null,
  literature_unpaywall_failed: () => null,
  hypothesis_draft_failed: () => ({ text: "One hypothesis draft could not be used", tone: "warn" }),
  report_contradiction_fixed: (d) => ({ text: `Report: corrected ${count(num(d.count), "statement")} that contradicted the results`, tone: "neutral" }),
  stage_interrupted: (_d, s) => ({ text: `${s} was interrupted and will continue where it stopped`, tone: "warn" }),
  commit_recovered: () => null,
  agent_log: (d, s) => ({ text: `${s ? `${s}: ` : ""}${str(d.message)}`, tone: "neutral" }),
  job_cancel_failed: (d) => ({ text: `Could not cancel job ${str(d.job)}: ${str(d.error)}`, tone: "warn" }),

  literature_phase_done: (d) => ({ text: `Literature: ${humanize(str(d.phase))} done`, tone: "neutral" }),
  hypotheses_ranked: (d) => ({ text: `Ranked ${count(num(d.eligible), "hypothesis", "hypotheses")}; offering the best ${(d.shortlist as unknown[] | undefined)?.length ?? 0}`, tone: "info" }),
  plan_repair: () => ({ text: "Plan checks found problems; the planner is fixing them", tone: "warn" }),
  resource_fetched: (d) => ({ text: `Downloaded ${str(d.name)} (${str(d.status)})`, tone: "info" }),
  experiment_submitted: (d) => ({
    text: `Submitted ${count(num(d.runs), "experiment run")} on ${str(d.partition)} (up to ${str(d.gpu_hours)} GPU-hours)`,
    tone: "info",
  }),
  experiment_runs_failed: (d) => ({ text: `${count(num(d.failed), "experiment run")} failed: ${str(d.summary)}`, tone: "warn" }),
  code_fix: (d) => ({ text: `Code: ${str(d.action)}`, tone: "neutral" }),
  review_returned_to_writer: () => ({ text: "Your comments went back to the writer", tone: "stale" }),
  episode_not_started: (d) => ({ text: `A worker could not start: ${str(d.error)}`, tone: "bad" }),
};

export function describeEvent(event: RunEvent): Described | null {
  const stage = event.stage ? stageTitle(event.stage) : "";
  const handler = DESCRIBE[event.kind];
  if (handler) return handler(event.data ?? {}, stage);
  return { text: `${stage ? `${stage}: ` : ""}${humanize(event.kind)}`, tone: "neutral" };
}
