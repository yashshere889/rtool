import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { CodeView, RunDetail, StageSummary } from "@/api/types";
import { fileDraftKey, num, pValue, stepStates } from "@/lib/code";

import { CodePage } from "./CodePage";

// CodeMirror needs a real browser layout; a text box stands in for it here
vi.mock("@/components/code/CodeEditor", () => ({
  default: ({ value, onChange, label, readOnly }: { value: string; onChange?: (v: string) => void; label: string; readOnly?: boolean }) => (
    <textarea aria-label={label} value={value} readOnly={readOnly} onChange={(e) => onChange?.(e.target.value)} />
  ),
  CodeDiff: ({ original, value }: { original: string; value: string }) => (
    <pre data-testid="diff">{`${original}\n---\n${value}`}</pre>
  ),
}));

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

const SOURCE = "def run(condition, seed, resources, smoke, workdir):\n    return {'acc': 1.0}\n";

const view = {
  stage: "code",
  status: "awaiting_gate",
  phase: "approve",
  code_root: "work/code/code",
  files: [
    { path: "experiment.py", size: 80, lines: 2, role: "experiment", editable: true },
    { path: "run.py", size: 900, lines: 40, role: "harness", editable: false },
  ],
  editable: true,
  why_not_editable: "",
  experiment_sha: "sha-1",
  generated_sha: "sha-1",
  tested_sha: "sha-1",
  plan_version: 1,
  conditions: ["base", "new"],
  baseline: "base",
  seeds: [0, 1],
  metrics: ["acc"],
  smoke: [
    { condition: "base", ok: true, seconds: 0.2, metrics: { acc: 1 }, error: "" },
    { condition: "new", ok: true, seconds: 0.3, metrics: { acc: 1 }, error: "" },
  ],
  fixes: [{ ts: new Date().toISOString(), phase: "smoke", signature: "missing_module", action: "added requirement numpy", detail: "" }],
  jobs: [],
  runs: [
    { condition: "base", seed: 0, status: "waiting", metrics: {}, seconds: null, error: "", log: null },
    { condition: "base", seed: 1, status: "waiting", metrics: {}, seconds: null, error: "", log: null },
    { condition: "new", seed: 0, status: "waiting", metrics: {}, seconds: null, error: "", log: null },
    { condition: "new", seed: 1, status: "waiting", metrics: {}, seconds: null, error: "", log: null },
  ],
  logs: [{ name: "logs/smoke-base.log", label: "Smoke test: base", size: 120 }],
  bundle_version: null,
  results_version: null,
} as unknown as CodeView;

describe("code helpers", () => {
  it("place the stage on its steps", () => {
    expect(stepStates({ status: "awaiting_gate", phase: "approve" })).toEqual(["done", "done", "attention", "waiting", "waiting"]);
    expect(stepStates({ status: "suspended", phase: "collect" })).toEqual(["done", "done", "done", "current", "waiting"]);
    expect(stepStates({ status: "failed", phase: "smoke" })[1]).toBe("failed");
    expect(stepStates({ status: "done", phase: "finish" })).toEqual(Array(5).fill("done"));
  });

  it("format numbers and name drafts like the server", () => {
    expect(fileDraftKey("code/experiment.py")).toBe("file_code_experiment_py");
    expect([num(3.14159), num(0.00001), num(null), pValue(0.0004), pValue(0.0213)]).toEqual(["3.142", "1.00e-5", "–", "< 0.001", "0.021"]);
  });
});

describe("CodePage", () => {
  it("edits experiment.py and sends it to be tested again", async () => {
    const fetch = vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      if (url.endsWith("/code")) return json(view);
      if (url.includes("/code/file")) return json({ path: "experiment.py", content: SOURCE, sha: "sha-1", editable: true });
      if (url.includes("/code/generated")) return json({ path: "experiment.py", content: SOURCE, sha: "sha-1", editable: false });
      if (url.includes("/drafts/") && method === "GET") return json({ detail: "no draft" }, 404);
      if (url.includes("/drafts/")) return json({ ...JSON.parse(String(init?.body)), saved_at: new Date().toISOString() });
      if (url.endsWith("/code/check")) {
        const content = JSON.parse(String(init?.body)).content as string;
        return json({ problems: content.includes("def run(") ? [] : ["experiment.py must define a top-level function run(...)"] });
      }
      if (url.endsWith("/actions")) return json({ action: { id: "a".repeat(32) }, kick: { submitted: false, detail: "ok" } }, 202);
      return json({ state: "queued" });
    });
    vi.stubGlobal("fetch", fetch);
    const stage = { id: "code", agent: "coder", status: "awaiting_gate", gate: null } as unknown as StageSummary;
    const run = { project: "sleep", run_id: "r1", stages: [stage], open_gates: [], queued_actions: [] } as unknown as RunDetail;
    render(wrap(<CodePage run={run} stage={stage} />));

    const steps = await screen.findByRole("list", { name: "Progress of this stage" });
    expect(within(steps).getAllByRole("listitem")[2]).toHaveAttribute("aria-current", "step");
    const editor = await screen.findByRole("textbox", { name: "experiment.py" });
    expect(screen.getByRole("tab", { name: "Tests & repairs (1)" })).toBeInTheDocument();

    await userEvent.clear(editor);
    await userEvent.type(editor, "print('no run function')");
    await screen.findByText(/must define a top-level function run/, {}, { timeout: 3000 });
    expect(screen.getByRole("button", { name: "Save and test again" })).toBeDisabled(); // the agent's checks fail

    await userEvent.clear(editor);
    await userEvent.type(editor, "def run(condition, seed, resources, smoke, workdir):{enter}    return {{'acc': 0.9}");
    const save = screen.getByRole("button", { name: "Save and test again" });
    await waitFor(() => expect(save).toBeEnabled(), { timeout: 3000 });
    await userEvent.click(save);
    await waitFor(() => expect(fetch.mock.calls.some(([u]) => String(u).endsWith("/actions"))).toBe(true));
    const [, init] = fetch.mock.calls.find(([u]) => String(u).endsWith("/actions"))!;
    expect(JSON.parse(String(init!.body))).toMatchObject({
      kind: "write_file",
      stage: "code",
      payload: { path: "code/experiment.py", base_sha: "sha-1", retest: true, content: expect.stringContaining("'acc': 0.9") },
    });

    await userEvent.click(screen.getByRole("button", { name: /Changes since the tool wrote it/ }));
    expect(await screen.findByTestId("diff")).toHaveTextContent("'acc': 0.9");
  });

  it("shows other files read-only and lists the runs", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.endsWith("/code")) return json({ ...view, status: "suspended", phase: "collect", editable: false, why_not_editable: "the experiment is running on the cluster" });
        if (url.includes("/code/file")) return json({ path: "run.py", content: "# harness\n", sha: "h", editable: false });
        if (url.includes("/drafts/")) return json({ detail: "no draft" }, 404);
        return json({ detail: "not here" }, 404);
      }),
    );
    const stage = { id: "code", agent: "coder", status: "suspended", gate: null, waiting_on: ["1000"] } as unknown as StageSummary;
    const run = { project: "sleep", run_id: "r1", stages: [stage], open_gates: [], queued_actions: [] } as unknown as RunDetail;
    render(wrap(<CodePage run={run} stage={stage} />));

    expect(await screen.findByText(/0 of 4 runs finished/)).toBeInTheDocument(); // running: the Runs tab comes first
    expect(screen.getByRole("button", { name: /Cancel the running jobs/ })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("tab", { name: "Code" }));
    await userEvent.click(screen.getByRole("button", { name: /run\.py/ }));
    expect(await screen.findByText(/rewrites it on every attempt/)).toBeInTheDocument();
    expect(await screen.findByRole("textbox", { name: "run.py" })).toHaveAttribute("readonly");
  });
});
