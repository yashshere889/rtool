import { describe, expect, it } from "vitest";

import { describeEvent } from "./events";

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
