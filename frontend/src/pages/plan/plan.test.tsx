import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ExperimentPlan, PlanView, RunDetail, StageSummary } from "@/api/types";
import { sameJson } from "@/lib/equal";
import { planChanges, resourceLink, seedList } from "@/lib/plan";

import { PlanPage } from "./PlanPage";

function wrap(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <QueryClientProvider client={client}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>
  );
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

afterEach(() => vi.unstubAllGlobals());

const plan = {
  hypothesis_id: "h1",
  hypothesis: "If clicks are phase-locked, recall improves",
  null_hypothesis: "No difference",
  objective: "Compare click timing",
  design: "Within-subject simulation",
  independent_variables: [{ name: "stimulation", description: "timing", values: ["sham", "phase-locked"] }],
  dependent_variables: [{ name: "recall gain", description: "word pairs", values: [] }],
  controls: ["same recordings"],
  conditions: [
    { name: "sham", description: "no clicks", is_baseline: true },
    { name: "phase_locked", description: "clicks at the up-state", is_baseline: false },
  ],
  ablations: [],
  metrics: [{ name: "recall_gain", definition: "word pairs gained", higher_is_better: true }],
  statistical_tests: [{ test: "paired t-test", comparison: "phase_locked vs sham", alpha: 0.05, correction: "none" }],
  seeds: [0, 1, 2],
  procedure: ["load", "simulate", "score"],
  resources: [
    {
      kind: "dataset",
      name: "Sleep-EDF",
      source: "huggingface",
      identifier: "demo-org/sleep-edf",
      subset: null,
      split: null,
      purpose: "recordings",
      verified: true,
      license: "cc-by-4.0",
      size_gb: 1.4,
      note: "",
    },
  ],
  compute: {
    partition: "gpu-l40s",
    gpu_type: "l40s",
    gpus_per_run: 1,
    time_limit: "00:39:00",
    runs: 6,
    hours_per_run: 0.5,
    gpu_hours_total: 3,
    gpu_memory_gb: 4,
    justification: "light",
  },
  expected_results: "phase-locked wins",
  risks: [],
  results_schema: {},
  issues: [],
  markdown_path: "work/plan/experiment_plan.md",
} as unknown as ExperimentPlan;

const view = {
  stage: "plan",
  version: 1,
  versions: [1],
  head: 1,
  plan,
  hypothesis_title: "Timing matters",
  by: null,
  edited_from: null,
  edit_note: "",
  created_at: new Date().toISOString(),
  editable: true,
  why_not_editable: "",
  cluster: {
    partitions: [
      { name: "gpu-l40s", gpu_type: "l40s", gpus_per_node: 2, gpu_memory_gb: 48, max_time: "3-00:00:00", preemptible: false, note: "", allowed: true },
    ],
    budget_gpu_hours: 40,
    min_seeds: 3,
    max_dataset_gb: 200,
  },
} as unknown as PlanView;

describe("plan helpers", () => {
  it("say in words what an edit changed, whatever the key order", () => {
    const after = structuredClone(plan);
    after.seeds = seedList(5);
    after.ablations = [{ name: "single", description: "one click", is_baseline: false }];
    after.compute = { ...after.compute, hours_per_run: 2 };
    expect(planChanges(plan, after)).toEqual([
      { label: "Ablations", detail: "1 added" },
      { label: "Seeds per condition", detail: "3 → 5" },
      { label: "Compute: hours per run", detail: "0.5 → 2" },
    ]);
    const reordered = JSON.parse(JSON.stringify(plan, Object.keys(plan).sort())) as ExperimentPlan;
    expect(sameJson({ a: 1, b: [1, { c: 2, d: 3 }] }, { b: [1, { d: 3, c: 2 }], a: 1 })).toBe(true);
    expect(sameJson(plan, { ...plan, seeds: [0] })).toBe(false);
    expect(planChanges(plan, { ...reordered, ...plan })).toEqual([]);
  });

  it("link resources to where they can be looked up", () => {
    expect(resourceLink({ source: "huggingface", kind: "dataset", identifier: "org/data" })).toBe("https://huggingface.co/datasets/org/data");
    expect(resourceLink({ source: "huggingface", kind: "model", identifier: "org/m" })).toBe("https://huggingface.co/org/m");
    expect(resourceLink({ source: "pip", kind: "code", identifier: "numpy>=2" })).toBe("https://pypi.org/project/numpy/");
    expect(resourceLink({ source: "url", kind: "other", identifier: "ftp://x" })).toBeNull();
  });
});

describe("PlanPage", () => {
  it("edits the plan, shows the checks, and sends the edit", async () => {
    const fetch = vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      if (url.endsWith("/plan") && method === "GET") return json(view);
      if (url.includes("/drafts/") && method === "GET") return json({ detail: "no draft" }, 404);
      if (url.includes("/drafts/")) return json({ ...JSON.parse(String(init?.body)), saved_at: new Date().toISOString() });
      if (url.endsWith("/plan/check")) {
        const value = JSON.parse(String(init?.body)).value as ExperimentPlan;
        const issues = value.seeds.length > 4 ? ["the plan needs about 60 GPU-hours; the budget is 40"] : [];
        return json({ ok: true, error: "", plan: { ...value, issues, compute: { ...value.compute, runs: 10, gpu_hours_total: 60 } }, changed: ["seeds"] });
      }
      if (url.endsWith("/impact")) return json([]);
      if (url.endsWith("/downloads")) return json({ started: false, items: [] });
      if (url.endsWith("/actions")) return json({ action: { id: "a".repeat(32) }, kick: { submitted: false, detail: "ok" } }, 202);
      return json({ state: "queued" });
    });
    vi.stubGlobal("fetch", fetch);
    const stage = { id: "plan", agent: "planner", status: "awaiting_gate", checkpoint: "experiment_plan", gate: null } as unknown as StageSummary;
    const run = { project: "sleep", run_id: "r1", stages: [stage, { id: "hypotheses", agent: "hypothesis" }], open_gates: [], queued_actions: [] } as unknown as RunDetail;
    render(wrap(<PlanPage run={run} stage={stage} />));

    await screen.findByText("Datasets, models and software");
    expect(screen.getByText("No problems found.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Edit the plan/ }));

    const seeds = await screen.findByRole("spinbutton", { name: /seeds\) per condition/ });
    await userEvent.clear(seeds);
    await userEvent.type(seeds, "5");
    expect(screen.getByText("Seeds per condition")).toBeInTheDocument();
    await screen.findByText(/budget is 40/, {}, { timeout: 3000 });
    expect(screen.getByText(/Worked out: 10 runs/)).toBeInTheDocument();

    await userEvent.type(screen.getByRole("textbox", { name: /Why \(optional/ }), "more power");
    const apply = screen.getByRole("button", { name: "Apply my edits" });
    await waitFor(() => expect(apply).toBeEnabled(), { timeout: 3000 });
    await userEvent.click(apply);
    await waitFor(() => expect(fetch.mock.calls.some(([u]) => String(u).endsWith("/actions"))).toBe(true));
    const [, init] = fetch.mock.calls.find(([u]) => String(u).endsWith("/actions"))!;
    const sent = JSON.parse(String(init!.body));
    expect(sent).toMatchObject({ kind: "edit", stage: "plan", payload: { artifact: "experiment_plan", base_version: 1, note: "more power" } });
    expect(sent.payload.value.seeds).toEqual([0, 1, 2, 3, 4]);
    await waitFor(() => expect(within(document.body).queryByRole("button", { name: "Apply my edits" })).toBeNull(), {
      timeout: 3000,
    }); // back to reading
  });
});
