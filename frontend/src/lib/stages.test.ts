import { describe, expect, it } from "vitest";

import type { RunDetail, StageSummary } from "@/api/types";

import { gateInfo, runHeadline, stageState, stageTitle } from "./stages";

function stage(id: string, status: StageSummary["status"], extra: Partial<StageSummary> = {}): StageSummary {
  return {
    id,
    agent: id,
    status,
    attempt: 1,
    checkpoint: null,
    needs: [],
    gate: null,
    stale_reason: null,
    last_error: null,
    waiting_on: [],
    started_at: null,
    finished_at: null,
    output_versions: {},
    usage: {},
    pending_feedback: 0,
    ...extra,
  };
}

function run(stages: StageSummary[], extra: Partial<RunDetail> = {}): RunDetail {
  return {
    project: "p",
    run_id: "r",
    topic: "t",
    pipeline: "research",
    pack: "generic",
    created_at: "2026-10-09T10:00:00+00:00",
    updated_at: null,
    last_outcome: "waiting_gate",
    stages,
    open_gates: [],
    error: null,
    tool_version: "0.1.0",
    run_state: null,
    queued_actions: [],
    worker: { active: false, detail: "", job_id: null, last_error: null },
    ...extra,
  };
}

describe("stage language", () => {
  it("names known stages and humanizes unknown ones", () => {
    expect(stageTitle("code")).toBe("Code & runs");
    expect(stageTitle("novelty_check")).toBe("Novelty check");
    expect(gateInfo("gpu_runs").approve).toBe("Approve GPU runs");
    expect(gateInfo("my_gate").title).toBe("My gate");
  });

  it("says what each status means for the researcher", () => {
    expect(stageState(stage("code", "suspended", { waiting_on: ["12", "13"] }))).toMatchObject({
      label: "Jobs running on the cluster",
      detail: "Slurm jobs 12, 13",
    });
    expect(stageState(stage("plan", "failed", { last_error: "boom" }))).toMatchObject({ tone: "bad", detail: "boom" });
    expect(stageState(stage("plan", "pending", { attempt: 0 })).label).toBe("Not started");
    const working = run([stage("literature", "done"), stage("plan", "pending", { attempt: 0 })], {
      worker: { active: true, detail: "working locally", job_id: null, last_error: null },
    });
    expect(stageState(working.stages[1], working).label).toBe("Working");
  });
});

describe("run headline", () => {
  it("puts a decision first, then failures, then progress", () => {
    const gate = { stage: "plan", gate: "experiment_plan", kind: "approve", in_stage: false, nonce: "n", opened_at: "", actions: ["approve", "reject"], summary: {} };
    const failed = run([stage("literature", "failed"), stage("plan", "awaiting_gate")], { open_gates: [gate] });
    expect(runHeadline(failed).text).toBe("Your decision needed: Approve the experiment plan");
    expect(runHeadline(run([stage("literature", "failed")])).text).toBe("Needs attention: Literature");
    expect(runHeadline(run([stage("literature", "done")])).text).toBe("Finished");
    expect(runHeadline(run([stage("code", "suspended")])).text).toBe("Experiments running on the cluster");
    expect(runHeadline(run([stage("literature", "pending", { attempt: 0 })], { last_outcome: "new" })).text).toBe(
      "Waiting to start",
    );
  });
});
