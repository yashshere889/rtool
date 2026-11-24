import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { CollectView, Protocol, ProtocolView, RunDetail, StageSummary } from "@/api/types";

import { CollectPage } from "./CollectPage";
import { ProtocolPage } from "./ProtocolPage";

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

const protocol = {
  unit: "participant",
  conditions: ["sham", "phase_locked"],
  baseline: "sham",
  primary_metric: "recall_gain",
  effect_size: 0.8,
  alpha: 0.05,
  power: 0.8,
  paired: false,
  n_per_condition: 26,
  allocation_path: "work/protocol/allocation.csv",
  materials: ["EEG headband", "word list"],
  blinding: "participants do not know which night had clicks",
  data_format: "data.csv: a CSV file with a header row",
  checklist: [
    { name: "a control condition", ok: true, detail: "sham" },
    { name: "blinding is stated", ok: true, detail: "" },
  ],
  markdown_path: "work/protocol/protocol.md",
} as unknown as Protocol;

const view = {
  stage: "protocol",
  version: 1,
  versions: [1],
  head: 1,
  protocol,
  plan: {
    objective: "Compare recall gain",
    design: "A parallel two-arm study",
    conditions: [
      { name: "sham", description: "no clicks", is_baseline: true },
      { name: "phase_locked", description: "clicks at the up-state", is_baseline: false },
    ],
    statistical_tests: [{ test: "Welch t-test", comparison: "phase_locked vs sham" }],
    procedure: ["recruit", "randomise", "measure"],
    risks: [],
  },
  hypothesis_title: "Timing matters",
  by: null,
  edited_from: null,
  edit_note: "",
  created_at: new Date().toISOString(),
  editable: true,
  why_not_editable: "",
  allocation_rows: 52,
  allocation_preview: [
    { unit: "U001", condition: "sham" },
    { unit: "U002", condition: "phase_locked" },
  ],
} as unknown as ProtocolView;

const stageOf = (id: string, agent: string, status = "awaiting_gate", gate: unknown = null) =>
  ({ id, agent, status, gate }) as unknown as StageSummary;
const runOf = (stages: StageSummary[]) =>
  ({ project: "sleep", run_id: "r1", stages, open_gates: [], queued_actions: [] }) as unknown as RunDetail;

describe("ProtocolPage", () => {
  it("shows the design and sample size, and sends an edit of the effect size", async () => {
    const fetch = vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      if (url.endsWith("/protocol") && method === "GET") return json(view);
      if (url.endsWith("/protocol/check")) {
        const value = JSON.parse(String(init?.body)).value as Protocol;
        return json({ ok: true, error: "", protocol: { ...value, n_per_condition: 64 }, changed: ["effect_size"] });
      }
      if (url.endsWith("/actions")) return json({ action: { id: "a".repeat(32) }, kick: { submitted: false, detail: "ok" } }, 202);
      return json({ events: [], cursor: 0 });
    });
    vi.stubGlobal("fetch", fetch);
    const stage = stageOf("protocol", "protocol");
    render(wrap(<ProtocolPage run={runOf([stage])} stage={stage} />));

    await screen.findByText(/26 participants per condition \(52 in all\)/);
    expect(screen.getByText("a control condition")).toBeInTheDocument();
    expect(screen.getByText("Welch t-test: phase_locked vs sham", { exact: false })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Edit the protocol/ }));
    const effect = await screen.findByRole("spinbutton", { name: /Smallest effect/ });
    await userEvent.clear(effect);
    await userEvent.type(effect, "0.5");
    await screen.findByText(/was 26/, {}, { timeout: 3000 });
    const apply = screen.getByRole("button", { name: "Apply my edits" });
    await waitFor(() => expect(apply).toBeEnabled(), { timeout: 3000 });
    await userEvent.click(apply);
    await waitFor(() => expect(fetch.mock.calls.some(([u]) => String(u).endsWith("/actions"))).toBe(true));
    const [, init] = fetch.mock.calls.find(([u]) => String(u).endsWith("/actions"))!;
    expect(JSON.parse(String(init!.body))).toMatchObject({
      kind: "edit",
      stage: "protocol",
      payload: { artifact: "protocol", base_version: 1, value: { effect_size: 0.5 } },
    });
  });
});

describe("CollectPage", () => {
  const collectView = {
    stage: "collect",
    status: "awaiting_gate",
    waiting: true,
    can_upload: true,
    why_not: "",
    columns: ["unit", "condition", "recall_gain"],
    unit: "participant",
    conditions: ["sham", "phase_locked"],
    n_per_condition: 26,
    paired: false,
    data_format: "data.csv: a CSV file",
    template: "unit,condition,recall_gain\nU001,sham,\n",
    inbox_file: false,
    received: null,
  } as unknown as CollectView;

  it("keeps a file the protocol rejects out, says why, and offers the checkpoint once data are in", async () => {
    let accepted = false;
    const fetch = vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      if (url.endsWith("/collect/inbox") && method === "PUT") {
        return accepted
          ? json({ rows: 52, units: { sham: 26, phase_locked: 26 } }, 201)
          : json({ detail: "data.csv does not match the protocol: line 2: condition 'coffee' is not one of sham, phase_locked; line 5: recall_gain is 'abc', not a number" }, 422);
      }
      if (url.endsWith("/collect")) return json(collectView);
      if (url.endsWith("/protocol")) return json(view);
      return json({ events: [], cursor: 0 });
    });
    vi.stubGlobal("fetch", fetch);
    const gate = { gate: "data_upload", stage: "collect", kind: "approve", nonce: "n1", actions: ["approve", "reject"], summary: { message: "Bring the data" } };
    const stage = stageOf("collect", "collect_data", "awaiting_gate", gate);
    render(wrap(<CollectPage run={runOf([stage])} stage={stage} />));

    await screen.findByText(/one row for each participant and condition: 26 participants per condition/);
    expect(screen.queryByRole("button", { name: "Analyse the data" })).toBeNull(); // nothing to approve yet
    const input = screen.getByLabelText("data.csv", { selector: "input" });
    const bad = new File(["unit,condition\n"], "data.csv", { type: "text/csv" });
    await userEvent.upload(input, bad);
    await userEvent.click(screen.getByRole("button", { name: /Upload and check/ }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("does not match the protocol");
    expect(alert).toHaveTextContent("condition 'coffee' is not one of sham, phase_locked");
    expect(alert).toHaveTextContent("recall_gain is 'abc', not a number");

    accepted = true;
    await userEvent.click(screen.getByRole("button", { name: /Upload and check/ }));
    await screen.findByText(/Received 52 rows/);
    expect(await screen.findByRole("button", { name: "Analyse the data" })).toBeInTheDocument(); // the checkpoint appears
    const sent = fetch.mock.calls.find(([u, i]) => String(u).endsWith("/collect/inbox") && (i as RequestInit)?.method === "PUT")!;
    expect((sent[1] as RequestInit).headers).toMatchObject({ "Content-Type": "text/csv", "X-Requested-With": "rtool" });
  });
});
