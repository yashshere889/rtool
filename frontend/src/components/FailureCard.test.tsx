import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { RunDetail, StageSummary } from "@/api/types";

import { FailureCard } from "./FailureCard";

const json = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });

const events = {
  cursor: 3,
  events: [
    { ts: "2026-10-21T03:00:00+00:00", run: "r1", episode: null, kind: "stage_retrying", stage: "code", data: { error: "x" } },
    {
      ts: "2026-10-21T03:31:00+00:00",
      run: "r1",
      episode: null,
      kind: "stage_failed",
      stage: "code",
      data: { attempt: 1, error: "KeyError: 'sections'", traceback: "Traceback ...\nKeyError: 'sections'" },
    },
  ],
};
const code = {
  editable: true,
  fixes: [
    { signature: "missing_import", phase: "smoke", action: "fix", detail: "", ts: "2026-10-21T02:00:00+00:00" },
    { signature: "missing_import", phase: "smoke", action: "fix", detail: "", ts: "2026-10-21T02:10:00+00:00" },
  ],
};

const run = { project: "sleep", run_id: "r1", queued_actions: [] } as unknown as RunDetail;
const stage = { id: "code", agent: "coder", status: "failed", last_error: "KeyError: 'sections'" } as unknown as StageSummary;

afterEach(() => vi.unstubAllGlobals());

describe("FailureCard", () => {
  it("says what was already tried, where to fix the code, and keeps the traceback to hand", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => Promise.resolve(json(url.includes("/events") ? events : code))),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter>
          <FailureCard run={run} stage={stage} />
        </MemoryRouter>
      </QueryClientProvider>,
    );

    expect(await screen.findByText(/tried again 1 time by itself/)).toBeInTheDocument();
    expect(await screen.findByText(/repaired the code 2 times on its own/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "edit it" })).toHaveAttribute("href", "/runs/sleep/r1/code?tab=code");
    expect(screen.getByText("and is waiting for you", { exact: false })).toBeInTheDocument();
    expect(screen.getByText("Technical details (for whoever maintains rtool)")).toBeInTheDocument();
    expect(screen.getByText(/Traceback \.\.\./)).toBeInTheDocument();
  });
});
