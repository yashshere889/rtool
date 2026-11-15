/** Turning the run's event log into an activity feed a researcher can read. */
import type { RunEvent } from "@/api/types";

import { STEPS as CODE_STEPS, stepStates, type StepState } from "./code";
import { count, humanize } from "./format";
import { LITERATURE_STEPS } from "./literature";
import { gateInfo, stageTitle, type Tone } from "./stages";

export interface Described {
  text: string;
  tone: Tone;
}

type Data = Record<string, unknown>;

const num = (value: unknown): number => (typeof value === "number" ? value : Number(value) || 0);
const str = (value: unknown): string => (value === undefined || value === null ? "" : String(value));

// the coding agent's phases; "approve" is shown by its checkpoint
const CODE_PHASES: Record<string, string> = {
  write: "writing the code",
  smoke: "testing the code on a CPU",
  submit: "submitting the runs to the cluster",
  collect: "collecting the results",
  finish: "analysing the results",
};

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

  literature_phase_done: () => null, // the progress strip shows the steps while they run
  literature_digested: (d) => ({
    text: `Summarised ${count(num(d.papers), "paper")}${num(d.failures) ? `; ${num(d.failures)} could not be summarised` : ""}`,
    tone: "info",
  }),
  literature_quality_warnings: (d) => ({
    text: `Literature quality checks to look at: ${((d.checks as string[] | undefined) ?? []).join(", ")}`,
    tone: "warn",
  }),
  phase_changed: (d, s) => (CODE_PHASES[str(d.phase)] ? { text: `${s}: ${CODE_PHASES[str(d.phase)]}`, tone: "neutral" } : null),
  smoke_result: (d) => ({
    text: `CPU test of ${str(d.condition)}${num(d.seed) ? ` (seed ${num(d.seed)})` : ""}: ${d.ok ? "passed" : `failed (${str(d.error)})`}`,
    tone: d.ok ? "ok" : "warn",
  }),
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

export interface Failure {
  ts: string;
  attempt: number;
  traceback: string;
  retries: number; // times the tool tried again by itself before giving up
}

/** A stage's latest failure, from the event log (null: none recorded). */
export function lastFailure(events: RunEvent[], stage: string): Failure | null {
  let failure: Failure | null = null;
  let retries = 0;
  for (const e of events) {
    if (e.stage !== stage) continue;
    if (e.kind === "stage_retrying") retries += 1;
    if (e.kind === "stage_failed") {
      failure = { ts: e.ts, attempt: num(e.data?.attempt), traceback: str(e.data?.traceback), retries };
      retries = 0;
    }
  }
  return failure;
}

export interface StageProgress {
  since: string | null; // when this attempt started (a resumed attempt keeps its start)
  steps: { label: string; state: StepState }[] | null; // null: this stage does not report steps
  latest: (Described & { ts: string }) | null; // the stage's newest activity
  queued: { since: string; job: string; expectedStart: string } | null; // a worker job waiting in the queue
}

/** Where the stage's current attempt starts in the log (a resumed attempt keeps its start), or -1. */
function attemptStart(events: RunEvent[], stage: string): number {
  return events.findLastIndex((e) => e.kind === "stage_started" && e.stage === stage && !e.data?.resumed);
}

/** Runs of equal neighbours as one item with how many there were (the activity feed's repeated lines). */
export function fold<T>(items: T[], key: (item: T) => string): { item: T; n: number }[] {
  const out: { item: T; n: number }[] = [];
  for (const item of items) {
    const last = out[out.length - 1];
    if (last && key(last.item) === key(item)) last.n += 1;
    else out.push({ item, n: 1 });
  }
  return out;
}

// what a stage reported along the way that the researcher should weigh before deciding at its checkpoint
const DECISION_WARNINGS = new Set([
  "literature_small_corpus",
  "literature_search_failed",
  "literature_quality_warnings",
  "hypothesis_draft_failed",
  "experiment_runs_failed",
]);

/** Warnings from the stage's current attempt, repeats counted ("A search on arxiv failed: ... (3 times)"). */
export function stageWarnings(events: RunEvent[], stage: string): string[] {
  const texts = events
    .slice(Math.max(attemptStart(events, stage), 0))
    .filter((e) => e.stage === stage && DECISION_WARNINGS.has(e.kind))
    .map((e) => describeEvent(e)?.text ?? "");
  const counts = new Map<string, number>();
  for (const t of texts) counts.set(t, (counts.get(t) ?? 0) + 1);
  return [...counts].map(([t, n]) => (n > 1 ? `${t} (${n} times)` : t));
}

/** How far a running stage has got, from the event log. */
export function stageProgress(events: RunEvent[], stage: string, agent: string): StageProgress {
  const start = attemptStart(events, stage);
  const mine = events.slice(Math.max(start, 0)).filter((e) => e.stage === stage);
  let steps: StageProgress["steps"] = null;
  if (agent === "literature") {
    const done = mine.filter((e) => e.kind === "literature_phase_done").map((e) => str(e.data?.phase));
    const reached = Math.max(-1, ...done.map((id) => LITERATURE_STEPS.findIndex((s) => s.id === id)));
    steps = LITERATURE_STEPS.map((s, i) => ({ label: s.label, state: i <= reached ? "done" : i === reached + 1 ? "current" : "waiting" }));
  } else if (agent === "coder") {
    const phase = mine.filter((e) => e.kind === "phase_changed").map((e) => str(e.data?.phase)).pop();
    const states = stepStates({ status: "running", phase: phase ?? "write" });
    steps = CODE_STEPS.map((s, i) => ({ label: s.label, state: states[i] }));
  }
  const latest = mine
    .map((e) => ({ e, d: describeEvent(e) }))
    .filter((x) => x.d !== null)
    .pop();
  const submitted = events.findLastIndex((e) => e.kind === "episode_submitted");
  const started = events.findLastIndex((e) => e.kind === "episode_started");
  const waiting = submitted > started ? events[submitted] : null;
  return {
    since: start >= 0 ? events[start].ts : null,
    steps,
    latest: latest ? { ...latest.d!, ts: latest.e.ts } : null,
    queued: waiting && { since: waiting.ts, job: str(waiting.data?.job), expectedStart: str(waiting.data?.expected_start) },
  };
}

/** Requests worth making at a stage's checkpoint, from what its current attempt reported. */
export function stageSuggestions(events: RunEvent[], stage: string): string[] {
  const out = new Set<string>();
  for (const e of events.slice(Math.max(attemptStart(events, stage), 0))) {
    if (e.stage !== stage) continue;
    if (e.kind === "literature_small_corpus")
      out.add(`Broaden the search: only ${num(e.data?.included)} relevant papers were found and the aim is ${str(e.data?.wanted)}.`);
    if (e.kind === "literature_search_failed") out.add(`Search ${str(e.data?.source)} again: it failed last time.`);
  }
  return [...out];
}
