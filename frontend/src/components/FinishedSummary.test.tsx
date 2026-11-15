import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { RunDetail } from "@/api/types";

import { FinishedSummary } from "./FinishedSummary";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

const comparison = (condition: string, extra: object) => ({
  metric: "recall_gain",
  condition,
  baseline: "sham",
  difference: 2.108,
  ci_low: 1.804,
  ci_high: 2.411,
  p_value: 0.0001,
  p_adjusted: 0.0002,
  significant: true,
  better: true,
  n: 4,
  constant_difference: false,
  ...extra,
});

const responses: Record<string, unknown> = {
  "/hypotheses": { selection: { title: "Phase-locked clicks beat untimed clicks", edited: false } },
  "/results": {
    metrics: [
      { name: "recall_gain", definition: "predicted recall gain", higher_is_better: true, primary: true },
      { name: "so_amplitude", definition: "amplitude", higher_is_better: true, primary: false },
    ],
    results: {
      complete: true,
      warnings: ["every seed gave the same value"],
      comparisons: [
        comparison("phase_locked", {}),
        comparison("fixed_interval", { difference: 0.1, significant: false, better: null }),
        comparison("phase_locked", { metric: "so_amplitude" }), // not the primary outcome
      ],
    },
  },
  "/review": { review: { verdict: "accept", summary: "A careful simulation study." } },
};

const run = {
  project: "sleep",
  run_id: "r1",
  stages: [
    { id: "hypotheses", agent: "hypothesis" },
    { id: "code", agent: "coder" },
    { id: "review", agent: "reviewer" },
  ],
} as unknown as RunDetail;

afterEach(() => vi.unstubAllGlobals());

describe("FinishedSummary", () => {
  it("states the hypothesis, the primary outcome against the baseline, and the review, from the artifacts", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => {
        const key = Object.keys(responses).find((k) => url.includes(k));
        return Promise.resolve(key ? json(responses[key]) : json({ detail: "not found" }, 404));
      }),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter>
          <FinishedSummary run={run} />
        </MemoryRouter>
      </QueryClientProvider>,
    );

    expect(await screen.findByRole("link", { name: "Phase-locked clicks beat untimed clicks" })).toHaveAttribute(
      "href",
      "/runs/sleep/r1/hypotheses",
    );
    expect(await screen.findByText("Better than baseline")).toBeInTheDocument();
    expect(screen.getByText("No clear difference")).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(2); // only the primary outcome's comparisons
    expect(screen.getByText(/1 warning about these results/)).toBeInTheDocument();
    expect(await screen.findByText(/recommended accepting the report, and you approved it/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Read the final report" })).toHaveAttribute("href", "/runs/sleep/r1/review?tab=final");
  });
});
