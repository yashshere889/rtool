import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { RunDetail } from "@/api/types";

import { ResultsTab } from "./ResultsTab";

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

const comparison = {
  baseline: "sham",
  ci_low: 1.8,
  ci_high: 2.4,
  test: "paired t-test",
  statistic: 9.1,
  effect_size: 11,
  n: 4,
  correction: "holm",
  no_seed_variation: false,
  constant_difference: false,
};

const resultsView = {
  version: 1,
  baseline: "sham",
  conditions: ["sham", "phase_locked", "fixed_interval"],
  metrics: [
    { name: "recall_gain", definition: "word pairs gained", higher_is_better: true, primary: true },
    { name: "so_amplitude", definition: "slow-oscillation amplitude", higher_is_better: true, primary: false },
  ],
  results: {
    runs: [],
    summaries: [],
    complete: true,
    results_path: "work/code/results.json",
    tables_path: "",
    figures: [],
    jobs: [],
    gpu_hours_requested: 2,
    code_fingerprint: "",
    notes: [],
    correction: "holm",
    warnings: ["all 3 conditions gave identical values for so_amplitude"],
    comparisons: [
      { ...comparison, metric: "recall_gain", condition: "phase_locked", difference: 2.1, p_value: 0.0004, p_adjusted: 0.0008, significant: true, better: true, analysis: "confirmatory" },
      { ...comparison, metric: "so_amplitude", condition: "phase_locked", difference: 9.5, p_value: null, p_adjusted: null, significant: false, better: null, constant_difference: true, analysis: "exploratory" },
    ],
  },
};

describe("ResultsTab", () => {
  it("marks the primary outcome, the comparisons it could not test, and the warnings", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => json(resultsView)),
    );
    const run = { project: "sleep", run_id: "r1" } as unknown as RunDetail;
    render(wrap(<ResultsTab run={run} />));

    expect(await screen.findByText("Primary outcome · confirmatory")).toBeInTheDocument();
    expect(screen.getByText("Exploratory")).toBeInTheDocument();
    expect(screen.getByRole("note", { name: "Warnings" })).toHaveTextContent("identical values for so_amplitude");

    const amplitude = screen.getByRole("heading", { name: "so_amplitude" }).closest("section")!;
    expect(within(amplitude).getByText("No test possible")).toBeInTheDocument();
    expect(within(amplitude).getByText(/identical for every seed/)).toBeInTheDocument();
    const recall = screen.getByRole("heading", { name: "recall_gain" }).closest("section")!;
    expect(within(recall).getByText("Better than baseline")).toBeInTheDocument();
    expect(within(recall).getByText(/p adjusted with Holm's method/)).toBeInTheDocument();
  });
});
