import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { HypothesesView, Hypothesis, RunDetail, StageSummary } from "@/api/types";
import { changedFields, draftProblems, toDraft } from "@/lib/hypotheses";

import { HypothesesPage } from "./HypothesesPage";

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

function hypothesis(id: string, title: string, extra: Partial<Hypothesis> = {}): Hypothesis {
  return {
    id,
    title,
    statement: `If ${title.toLowerCase()}, recall improves`,
    null_hypothesis: "No difference",
    rationale: "Earlier studies suggest it.",
    evidence: [{ paper_id: "doi:1", quote: "Recall was higher after stimulation.", verified: true, support: 0.97 }],
    independent_variables: ["timing"],
    dependent_variables: ["recall"],
    controls: [],
    datasets: ["Sleep-EDF"],
    methods: ["closed-loop stimulation"],
    success_criteria: "A significant gain",
    risks: ["simulated recall"],
    compute: "small_gpu",
    source_opportunity: "gap-1",
    related_paper_ids: ["doi:1"],
    novelty: { novel: true, closest_paper_id: "doi:1", overlap: "Differs in timing.", score: 7 },
    feasibility_notes: [],
    scores: { novelty: 4, feasibility: 5, testability: 5, significance: 4, clarity: 5 },
    total_score: 0.89,
    rubric_comments: ["A clean comparison."],
    rejected_reason: null,
    ...extra,
  } as Hypothesis;
}

const paper = {
  id: "doi:1",
  bib_key: "morgan2019",
  title: "Phase-locked stimulation",
  authors: ["A. Morgan", "L. Chen"],
  year: 2019,
  venue: "Neuron",
  citation_count: 812,
  links: { doi: "https://doi.org/10.5555/1", arxiv: null, openalex: null, landing: null, open_access: [] },
};

function view(selection: HypothesesView["selection"] = null): HypothesesView {
  const titles = ["Timing matters", "Thresholds adapt", "Coupling explains"];
  return {
    stage: "hypotheses",
    version: 1,
    versions: [1],
    topic: "Sleep and memory",
    shortlist: titles.map((title, index) => ({
      index,
      hypothesis: hypothesis(`h${index + 1}`, title),
      papers: [{ paper, roles: ["evidence", "closest"] }],
      bibtex: "@article{morgan2019,}",
      gpu_hours: 8,
    })),
    others: [
      { index: null, hypothesis: hypothesis("h9", "Copy", { rejected_reason: "near-duplicate of h1" }), papers: [], bibtex: "", gpu_hours: 8 },
    ],
    selection,
    weights: { novelty: 0.3, feasibility: 0.25, testability: 0.2, significance: 0.15, clarity: 0.1 },
    budget_gpu_hours: null,
    markdown: "work/hypotheses/hypotheses.md",
  } as unknown as HypothesesView;
}

const gate = {
  stage: "hypotheses",
  gate: "hypothesis_selection",
  kind: "select",
  in_stage: false,
  nonce: "nonce1",
  opened_at: "",
  actions: ["select", "reject"],
  summary: { artifacts: { hypothesis_set: 1 }, options: [] },
};

function setup(stage: Partial<StageSummary>, body: HypothesesView) {
  const fetch = vi.fn(async (url: string, init?: RequestInit) => {
    if (url.endsWith("/hypotheses")) return json(body);
    if (url.endsWith("/impact")) return json([{ stage: "plan", status: "done", effect: "redo", detail: "" }]);
    if (init?.method === "POST") return json({ action: { id: "a".repeat(32) }, kick: { submitted: false, detail: "ok" } }, 202);
    return json({ state: "queued" });
  });
  vi.stubGlobal("fetch", fetch);
  const full = { id: "hypotheses", agent: "hypothesis", status: "awaiting_gate", checkpoint: "hypothesis_selection", gate: null, ...stage } as StageSummary;
  const run = { project: "sleep", run_id: "r1", stages: [full], open_gates: full.gate ? [full.gate] : [], queued_actions: [] } as unknown as RunDetail;
  render(wrap(<HypothesesPage run={run} stage={full} />));
  return fetch;
}

describe("editing a hypothesis", () => {
  it("sends only what changed, lists one item per line", () => {
    const h = hypothesis("h1", "Timing matters");
    const draft = { ...toDraft(h), statement: "A new claim", datasets: "MASS\n\n Sleep-EDF " };
    expect(changedFields(h, draft)).toEqual({ statement: "A new claim", datasets: ["MASS", "Sleep-EDF"] });
    expect(changedFields(h, toDraft(h))).toEqual({});
    expect(draftProblems({ ...toDraft(h), statement: " ", dependent_variables: "" })).toEqual([
      "Hypothesis cannot be empty.",
      "Name at least one dependent variable.",
    ]);
  });
});

describe("HypothesesPage", () => {
  it("shows three hypotheses with their papers and citations, and chooses an edited one", async () => {
    const fetch = setup({ gate }, view());
    const columns = await screen.findAllByRole("article");
    expect(columns).toHaveLength(3);
    const second = columns[1];
    expect(within(second).getByRole("heading", { name: "Thresholds adapt" })).toBeInTheDocument();
    expect(within(second).getByText("Quoted as evidence")).toBeInTheDocument();
    expect(within(second).getByRole("link", { name: /Show in paper/ })).toHaveAttribute(
      "href",
      "/runs/sleep/r1/hypotheses/reader?paper=doi%3A1&quote=Recall+was+higher+after+stimulation.",
    );
    const opens = within(second).getAllByRole("link", { name: "Phase-locked stimulation" }); // closest work, papers used
    expect(opens.map((a) => a.getAttribute("href"))).toEqual(Array(2).fill("/runs/sleep/r1/hypotheses/reader?paper=doi%3A1"));
    expect(within(second).getByText(/Morgan, L. Chen \(2019\)\. Phase-locked stimulation\. Neuron\./)).toBeInTheDocument();

    await userEvent.click(within(second).getByRole("button", { name: /Edit & choose/ }));
    const statement = screen.getByRole("textbox", { name: /^Hypothesis/ });
    await userEvent.clear(statement);
    await userEvent.type(statement, "If the threshold adapts, older adults respond like the young");
    await userEvent.type(screen.getByRole("textbox", { name: /note for the planner/i }), "prefer public data");
    expect(screen.getByText("You changed: Hypothesis.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Choose my version" }));

    await waitFor(() => expect(fetch.mock.calls.some(([, init]) => init?.method === "POST")).toBe(true));
    const [url, init] = fetch.mock.calls.find(([, i]) => i?.method === "POST")!;
    expect(url).toBe("/api/runs/sleep/r1/actions");
    expect(JSON.parse(String(init!.body))).toEqual({
      kind: "select",
      stage: "hypotheses",
      gate: "hypothesis_selection",
      payload: {
        index: 1,
        note: "prefer public data",
        edit: { statement: "If the threshold adapts, older adults respond like the young" },
        expect: { gate_nonce: "nonce1", artifacts: { hypothesis_set: 1 } },
      },
    });
  });

  it("shows the choice with your edits, and can reopen it after showing the impact", async () => {
    const selection = {
      version: 1,
      index: 0,
      title: "Timing matters",
      edited: true,
      changes: [{ field: "datasets", before: ["Sleep-EDF"], after: ["MASS"] }],
      note: "",
      from_version: 1,
      by: "yash (web)",
      created_at: new Date().toISOString(),
    };
    const fetch = setup({ status: "done" }, view(selection));
    const banner = await screen.findByRole("region", { name: "Your choice" });
    expect(within(banner).getByText(/Chosen by yash .* with your edits to Datasets/)).toBeInTheDocument();
    const [first] = screen.getAllByRole("article");
    expect(within(first).getByText("Your choice")).toBeInTheDocument();
    expect(within(first).getByText("was: Sleep-EDF")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Choose this one" })).toBeNull();

    await userEvent.click(within(banner).getByRole("button", { name: /Change my choice/ }));
    await screen.findByText(/would be redone/);
    await userEvent.click(screen.getByRole("button", { name: "Reopen the checkpoint" }));
    await waitFor(() => expect(fetch.mock.calls.some(([, init]) => init?.method === "POST")).toBe(true));
    const [, init] = fetch.mock.calls.find(([, i]) => i?.method === "POST")!;
    expect(JSON.parse(String(init!.body))).toMatchObject({ kind: "reopen", stage: "hypotheses" });
  });
});
