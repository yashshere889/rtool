import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { RunDetail } from "@/api/types";

import { RunTools } from "./RunTools";
import { ValidityBadge } from "./ValidityBadge";

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

const run = {
  project: "sleep",
  run_id: "r1",
  stages: [
    { id: "literature", agent: "literature" },
    { id: "plan", agent: "planner" },
  ],
} as unknown as RunDetail;

describe("ValidityBadge", () => {
  it("says a field with no card has no level, and what a measured field was measured on", async () => {
    const note = { pack: "generic", measured: false, text: "No validity card exists for the generic field on this server, so no level is claimed.", runs: 0 };
    vi.stubGlobal("fetch", vi.fn(async () => json(note)));
    const first = render(wrap(<ValidityBadge run={run} />));
    expect(await screen.findByText("Not measured.")).toBeInTheDocument();
    expect(screen.getByText(/so no level is claimed/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "See the card" })).toHaveAttribute("href", "/settings?tab=validity");
    first.unmount();

    vi.stubGlobal("fetch", vi.fn(async () => json({ ...note, measured: true, text: "Measured on 5 scored benchmark runs of the generic field (model Qwen, code abc)." })));
    render(wrap(<ValidityBadge run={run} />));
    expect(await screen.findByText("Measured field.")).toBeInTheDocument();
    expect(screen.getByText(/Measured on 5 scored benchmark runs/)).toBeInTheDocument();
  });
});

describe("RunTools", () => {
  it("makes a copy with a named stage, and shows why it was refused", async () => {
    const fetch = vi.fn(async (url: string, init?: RequestInit) =>
      url.endsWith("/replay") && init?.method === "POST"
        ? json({ detail: "run directory already exists: /runs/sleep/r1-plan-again" }, 409)
        : json({}),
    );
    vi.stubGlobal("fetch", fetch);
    render(wrap(<RunTools run={run} />));
    await userEvent.click(screen.getByRole("button", { name: "Redo a stage in a copy" }));
    await userEvent.selectOptions(await screen.findByLabelText("Stage to redo"), "plan");
    expect(screen.getByText("The copy is called r1-plan-again.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Make the copy" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("already exists");
    const [, init] = fetch.mock.calls.find(([u]) => String(u).endsWith("/replay"))!;
    expect(JSON.parse(String(init!.body))).toEqual({ stage: "plan", tag: "again" });
  });

  it("archives with a lowered cap and shows what was left out; explains when no folder is set", async () => {
    let state = "none";
    const view = (extra: object = {}) => ({
      can_archive: true, why_not: "", run_bytes: 12_000_000, default_max_file_mb: 200, state, path: "", files: 0, bytes: 0, skipped: [], error: "", ...extra,
    });
    const fetch = vi.fn(async (url: string, init?: RequestInit) => {
      if (url.endsWith("/archive") && init?.method === "POST") {
        state = "done";
        return json(view({ path: "/data/sleep-r1.tar.gz", files: 40, bytes: 11_000_000, skipped: [{ path: "work/big.bin", bytes: 90_000_000 }] }), 202);
      }
      return json(view());
    });
    vi.stubGlobal("fetch", fetch);
    render(wrap(<RunTools run={run} />));
    await userEvent.click(screen.getByRole("button", { name: "Archive this run" }));
    expect(await screen.findByText(/About 12 MB would be packed/)).toBeInTheDocument();
    const cap = screen.getByLabelText(/Leave out single files larger than/);
    await userEvent.clear(cap);
    await userEvent.type(cap, "50");
    await userEvent.click(screen.getByRole("button", { name: "Archive" }));
    expect(await screen.findByText(/Archived 40 files/)).toBeInTheDocument();
    expect(screen.getByText("/data/sleep-r1.tar.gz")).toBeInTheDocument();
    expect(screen.getByText("work/big.bin")).toBeInTheDocument();
    const [, init] = fetch.mock.calls.find(([u, i]) => String(u).endsWith("/archive") && (i as RequestInit)?.method === "POST")!;
    expect(JSON.parse(String(init!.body))).toEqual({ max_file_mb: 50 });
  });

  it("is disabled, with the reason, when no archive folder is set", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => json({ can_archive: false, why_not: "no archive folder is set: set paths.archive_root in the configuration", run_bytes: 1, default_max_file_mb: 200, state: "none", path: "", files: 0, bytes: 0, skipped: [], error: "" })),
    );
    render(wrap(<RunTools run={run} />));
    await userEvent.click(screen.getByRole("button", { name: "Archive this run" }));
    expect(await screen.findByText(/set paths.archive_root/)).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole("button", { name: "Archive" })).toBeNull());
  });
});
