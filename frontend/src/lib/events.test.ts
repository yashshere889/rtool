import { describe, expect, it } from "vitest";

import { describeEvent, lastFailure, stageProgress } from "./events";

const event = (kind: string, data: Record<string, unknown> = {}, stage?: string) => ({
  ts: "2026-10-09T10:00:00+00:00",
  run: "r",
  episode: null,
  kind,
  stage,
  data,
});

describe("activity feed wording", () => {
  it("turns engine events into sentences", () => {
    expect(describeEvent(event("gate_opened", { gate: "hypothesis_selection" }, "hypotheses"))?.text).toBe(
      "Your decision needed: Choose a hypothesis",
    );
    expect(describeEvent(event("gate_opened", { gate: "hypothesis_selection", reopened: true }))?.text).toBe(
      "Reopened at your request: Choose a hypothesis",
    );
    expect(describeEvent(event("stage_finished", {}, "code"))?.text).toBe("Code & runs finished");
    expect(describeEvent(event("literature_downloaded", { downloaded: 9, unavailable: 1 }))?.text).toBe(
      "Downloaded 9 papers; 1 had no open-access copy",
    );
    expect(describeEvent(event("action_rejected", { action: "approve", error: "too late" }))).toMatchObject({
      tone: "bad",
      text: "Your decision (approve) was refused: too late",
    });
  });

  it("hides bookkeeping and still shows unknown events", () => {
    expect(describeEvent(event("artifact_committed"))).toBeNull();
    expect(describeEvent(event("novelty_checked", {}, "novelty"))?.text).toBe("Novelty: Novelty checked");
  });
});

describe("lastFailure", () => {
  it("returns the latest failure with only the automatic retries that led to it", () => {
    const events = [
      { ...event("stage_retrying", {}, "code") },
      { ...event("stage_failed", { attempt: 1, traceback: "old" }, "code") },
      { ...event("stage_retrying", {}, "code") },
      { ...event("stage_retrying", {}, "plan") }, // another stage
      { ...event("stage_retrying", {}, "code") },
      { ...event("stage_failed", { attempt: 2, traceback: "KeyError: 'sections'" }, "code"), ts: "2026-10-21T03:31:00+00:00" },
    ];
    expect(lastFailure(events, "code")).toEqual({
      ts: "2026-10-21T03:31:00+00:00",
      attempt: 2,
      traceback: "KeyError: 'sections'",
      retries: 2,
    });
    expect(lastFailure(events, "report")).toBeNull();
  });
});

describe("stageProgress", () => {
  const at = (kind: string, data: Record<string, unknown> = {}, stage?: string, ts = "2026-10-09T10:00:00+00:00") => ({
    ...event(kind, data, stage),
    ts,
  });

  it("follows the literature steps of the current attempt, and a worker still in the queue", () => {
    const events = [
      at("stage_started", { resumed: false }, "literature"),
      at("literature_phase_done", { phase: "plan" }, "literature"),
      at("literature_phase_done", { phase: "search" }, "literature"),
      at("literature_phase_done", { phase: "screen" }, "literature"), // an earlier attempt
      at("stage_started", { resumed: false }, "literature", "2026-10-09T11:00:00+00:00"),
      at("literature_phase_done", { phase: "plan" }, "literature"),
      at("literature_phase_done", { phase: "search" }, "literature", "2026-10-09T11:05:00+00:00"),
      at("stage_started", { resumed: true }, "literature"), // a resumed attempt keeps its steps
      at("episode_started"),
      at("episode_submitted", { job: "123", expected_start: "2026-10-09T12:00:00" }, undefined, "2026-10-09T11:10:00+00:00"),
    ];
    const progress = stageProgress(events, "literature", "literature");
    expect(progress.since).toBe("2026-10-09T11:00:00+00:00");
    expect(progress.steps?.slice(0, 4).map((s) => s.state)).toEqual(["done", "done", "current", "waiting"]);
    expect(progress.latest?.text).toBe("Literature started (continuing)");
    expect(progress.queued).toEqual({ since: "2026-10-09T11:10:00+00:00", job: "123", expectedStart: "2026-10-09T12:00:00" });
  });

  it("uses the coding steps, and reports no steps for stages that have none", () => {
    const events = [at("stage_started", {}, "code"), at("phase_changed", { phase: "smoke" }, "code"), at("episode_started")];
    expect(stageProgress(events, "code", "coder").steps?.map((s) => s.state)).toEqual(["done", "current", "waiting", "waiting", "waiting"]);
    expect(stageProgress(events, "code", "coder").queued).toBeNull();
    expect(stageProgress([at("stage_started", {}, "plan")], "plan", "planner").steps).toBeNull();
  });
});
