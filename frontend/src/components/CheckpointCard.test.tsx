import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { OpenGate, RunDetail } from "@/api/types";
import { usePaperMarks } from "@/lib/paperMarks";

import { CheckpointCard } from "./CheckpointCard";

const gate: OpenGate = {
  stage: "hypotheses",
  gate: "hypothesis_selection",
  kind: "select",
  in_stage: false,
  nonce: "abc123",
  opened_at: "2026-10-09T10:00:00+00:00",
  actions: ["select", "reject"],
  summary: {
    prompt: "Choose the hypothesis to test.",
    artifacts: { hypothesis_set: 2 },
    review_files: ["work/hypotheses/hypotheses.md"],
    options: [
      { index: 0, title: "Phase-locked clicks beat untimed clicks" },
      { index: 1, title: "Adaptive thresholds in ageing" },
    ],
  },
};

const run = {
  project: "sleep",
  run_id: "r1",
  queued_actions: [],
} as unknown as RunDetail;

function wrap(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <QueryClientProvider client={client}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>
  );
}

afterEach(() => vi.unstubAllGlobals());

/** The decision sent (the card also loads the run's events). */
const actionCall = (fetch: ReturnType<typeof vi.fn>) => fetch.mock.calls.find(([url]) => String(url).endsWith("/actions"));

describe("CheckpointCard", () => {
  it("sends the chosen option, bound to the gate the researcher saw", async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ action: { id: "x".repeat(32) }, kick: { submitted: true, detail: "working" } }), {
        status: 202,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetch);
    render(wrap(<CheckpointCard run={run} gate={gate} />));

    const choose = screen.getByRole("button", { name: "Choose" });
    expect(choose).toBeDisabled(); // nothing chosen yet
    expect(screen.getByRole("link", { name: /hypotheses.md/ })).toHaveAttribute(
      "href",
      "/api/runs/sleep/r1/files/work/hypotheses/hypotheses.md",
    );
    await userEvent.click(screen.getByLabelText(/Adaptive thresholds/));
    await userEvent.click(choose);

    await waitFor(() => expect(actionCall(fetch)).toBeDefined());
    const [url, init] = actionCall(fetch)!;
    expect(url).toBe("/api/runs/sleep/r1/actions");
    expect(init.headers["X-Requested-With"]).toBe("rtool");
    expect(JSON.parse(init.body)).toEqual({
      kind: "select",
      stage: "hypotheses",
      gate: "hypothesis_selection",
      payload: { index: 1, expect: { gate_nonce: "abc123", artifacts: { hypothesis_set: 2 } } },
    });
  });

  it("asks for feedback before requesting changes", async () => {
    const fetch = vi.fn().mockResolvedValue(new Response("{}", { status: 202 }));
    vi.stubGlobal("fetch", fetch);
    render(wrap(<CheckpointCard run={run} gate={{ ...gate, kind: "approve", gate: "literature_review" }} />));
    await userEvent.click(screen.getByRole("button", { name: "Request changes" }));
    const send = screen.getByRole("button", { name: "Send" });
    expect(send).toBeDisabled();
    await userEvent.type(screen.getByLabelText("What should change?"), "Drop the modelling papers");
    await userEvent.click(send);
    await waitFor(() => expect(actionCall(fetch)).toBeDefined());
    expect(JSON.parse(actionCall(fetch)![1].body).payload.feedback).toBe("Drop the modelling papers");
  });

  it("can keep the current choice at a reopened selection", async () => {
    const fetch = vi.fn().mockResolvedValue(new Response("{}", { status: 202 }));
    vi.stubGlobal("fetch", fetch);
    const reopened = {
      ...gate,
      actions: ["select", "approve", "reject"],
      summary: { ...gate.summary, reopened: true, current: { index: 1, version: 3 } },
    };
    render(wrap(<CheckpointCard run={run} gate={reopened} />));
    expect(screen.getByText(/You reopened this checkpoint\. Keep your current choice \(option 2\)/)).toBeInTheDocument();
    expect(screen.getByText("(your current choice)")).toBeInTheDocument();
    expect(screen.queryByText(/Reopened/)).not.toBeInTheDocument(); // not listed again among the details
    await userEvent.click(screen.getByRole("button", { name: "Keep my current choice" }));
    await waitFor(() => expect(actionCall(fetch)).toBeDefined());
    expect(JSON.parse(actionCall(fetch)![1].body)).toMatchObject({ kind: "approve", gate: "hypothesis_selection" });
  });

  it("will not approve over unsent Keep/Drop marks, which outlive the page that made them", async () => {
    const litGate: OpenGate = { ...gate, stage: "literature", kind: "approve", gate: "literature_review", nonce: "lit1", actions: ["approve", "reject"] };
    function Marker() {
      const { mark } = usePaperMarks("sleep", "r1", "lit1");
      return <button onClick={() => mark("doi:1", "exclude")}>mark</button>;
    }
    const first = render(wrap(<Marker />));
    await userEvent.click(screen.getByRole("button", { name: "mark" }));
    first.unmount(); // e.g. opening a paper in the reader

    render(wrap(<CheckpointCard run={run} gate={litGate} />));
    const approve = screen.getByRole("button", { name: /Approve/ });
    expect(approve).toBeDisabled();
    expect(screen.getByRole("note")).toHaveTextContent("You marked 1 paper Keep or Drop");
    await userEvent.click(screen.getByRole("button", { name: "Discard marks" }));
    expect(approve).toBeEnabled();
    expect(sessionStorage.length).toBe(0);
  });

  it("adds what the stage warned about to the warnings to check", async () => {
    const events = {
      cursor: 2,
      events: [
        { ts: "2026-10-09T10:00:00+00:00", run: "r1", episode: null, kind: "stage_started", stage: "literature", data: {} },
        { ts: "2026-10-09T10:01:00+00:00", run: "r1", episode: null, kind: "literature_small_corpus", stage: "literature", data: { included: 9, wanted: 30 } },
      ],
    };
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(new Response(JSON.stringify(events), { status: 200, headers: { "content-type": "application/json" } }))));
    const litGate: OpenGate = { ...gate, stage: "literature", kind: "approve", gate: "literature_review", nonce: "lit2", actions: ["approve", "reject"] };
    render(wrap(<CheckpointCard run={run} gate={litGate} />));
    expect(await screen.findByRole("note", { name: "Warnings" })).toHaveTextContent("Only 9 relevant papers found (aimed for 30)");
  });

  it("offers undo instead of buttons while a decision is queued", () => {
    const queued = { ...run, queued_actions: [{ id: "q1", kind: "approve", stage: "hypotheses", gate: "hypothesis_selection" }] };
    render(wrap(<CheckpointCard run={queued as unknown as RunDetail} gate={gate} />));
    expect(screen.getByRole("button", { name: /Undo/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Choose" })).not.toBeInTheDocument();
  });

  const gpuGate: OpenGate = {
    stage: "code",
    gate: "gpu_runs",
    kind: "approve",
    in_stage: true,
    nonce: "def456",
    opened_at: "2026-10-29T10:00:00+00:00",
    actions: ["approve", "reject"],
    summary: { message: "Smoke tests passed.", runs: 12, warnings: [], smoke: ["sham ok in 0.1s", "phase_locked ok in 0.1s"] },
  };

  it("leaves out empty facts and shows lists as lists", () => {
    render(wrap(<CheckpointCard run={run} gate={gpuGate} />));
    expect(screen.queryByText("[]")).not.toBeInTheDocument();
    expect(screen.queryByRole("note", { name: "Warnings" })).not.toBeInTheDocument();
    expect(screen.getByText("sham ok in 0.1s").tagName).toBe("LI");
  });

  it("puts warnings first, as a list to check", () => {
    const warned = { ...gpuGate, summary: { ...gpuGate.summary, warnings: ["every seed gave the same value", "the condition was ignored"] } };
    render(wrap(<CheckpointCard run={run} gate={warned} />));
    const note = screen.getByRole("note", { name: "Warnings" });
    expect(note).toHaveTextContent("every seed gave the same value");
    expect(note.querySelectorAll("li")).toHaveLength(2);
    expect(screen.queryByText(/\["every seed/)).not.toBeInTheDocument();
  });
});
