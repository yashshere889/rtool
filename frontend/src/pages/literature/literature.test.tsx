import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { LiteratureView, OpenGate, PaperDetail, RunDetail } from "@/api/types";

import { ChoicesBar } from "./LiteraturePage";
import { PaperPanel } from "./PaperPanels";
import { ScopeCard } from "./ScopeCard";
import { UploadDialog } from "./UploadDialog";

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

const gate = {
  stage: "literature",
  gate: "literature_review",
  kind: "approve",
  in_stage: false,
  nonce: "n1",
  opened_at: "",
  actions: ["approve", "reject"],
  summary: { artifacts: { corpus: 1 } },
};
const run = { project: "sleep", run_id: "r1", open_gates: [gate], queued_actions: [] } as unknown as RunDetail;

describe("UploadDialog", () => {
  it("uploads the file into the inbox and can have the stage read it now", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(json({ file: "work/literature/inbox/doi-1.pdf", paper_id: "doi:1", title: "Paywalled" }, 201))
      .mockResolvedValueOnce(json({ action: { id: "a".repeat(32) }, kick: { submitted: true, detail: "ok" } }, 202));
    vi.stubGlobal("fetch", fetch);
    render(wrap(<UploadDialog run={run} stage="literature" open onOpenChange={() => {}} paperId="doi:1" initialTitle="Paywalled" />));

    const file = new File(["%PDF-1.4 my copy"], "paper.pdf", { type: "application/pdf" });
    await userEvent.upload(screen.getByLabelText("File (PDF or HTML)"), file);
    await userEvent.click(screen.getByRole("button", { name: /Upload/ }));

    await screen.findByText(/waiting in the literature agent's inbox/);
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe("/api/runs/sleep/r1/literature/inbox?title=Paywalled&paper_id=doi%3A1");
    expect(init.method).toBe("PUT");
    expect(init.headers).toMatchObject({ "X-Requested-With": "rtool", "Content-Type": "application/pdf" });
    expect(init.body).toBe(file);

    await userEvent.click(screen.getByRole("button", { name: "Read it now" }));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
    const sent = JSON.parse(fetch.mock.calls[1][1].body);
    expect(sent).toMatchObject({ kind: "reject", stage: "literature", gate: "literature_review" });
    expect(sent.payload.feedback).toContain("Paywalled");
  });
});

describe("paper panels", () => {
  const detail = {
    paper: { id: "doi:1", read: "full_text", links: {}, authors: ["A. Morgan"], title: "T", year: 2019 },
    problem: "Does sound help memory?",
    approach: "Clicks during sleep.",
    methods: ["closed-loop stimulation"],
    datasets: [],
    metrics: [],
    claims: [
      { index: 0, kind: "result", text: "Recall improved", quote: "Recall was higher.", verified: true, support: 1, method: "", dataset: "", metric: "recall", value: "2.3" },
      { index: 1, kind: "limitation", text: "Small sample", quote: "The sample was small.", verified: false, support: 0.4, method: "", dataset: "", metric: "", value: "" },
      { index: 2, kind: "result", text: "Recall rose by 73%", quote: "Recall was higher.", verified: false, support: 1, method: "", dataset: "", metric: "", value: "", unquoted_numbers: ["73"] },
      { index: 3, kind: "result", text: "64 adults improved", quote: "Recall was higher.", verified: true, support: 1, method: "", dataset: "", metric: "", value: "", unquoted_numbers: ["64"] },
    ],
    methods_section: "",
    section_titles: [],
    gaps: [],
    open_problems: [],
    coverage: [],
    cites: [],
    cited_by: [],
    bibtex: "",
  } as unknown as PaperDetail;
  const ctx = { project: "sleep", runId: "r1", stage: "literature" };

  it("shows findings with their verification and a link into the paper", () => {
    render(wrap(<PaperPanel kind="findings" detail={detail} ctx={ctx} />));
    expect(screen.getByText("Recall improved")).toBeInTheDocument();
    expect(screen.getAllByText(/Quote found in the paper \(100% match\)/)).toHaveLength(2); // claims 0 and 3
    expect(screen.getByText("recall = 2.3")).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: /Show in paper/ })[0]).toHaveAttribute(
      "href",
      "/runs/sleep/r1/literature/reader?paper=doi%3A1&claim=0",
    );
  });

  it("says when a real quote carries a number the paper does not contain", () => {
    render(wrap(<PaperPanel kind="findings" detail={detail} ctx={ctx} />));
    expect(screen.getByText(/Number not in the paper: 73/)).toBeInTheDocument();
    expect(screen.getByText("Not in this quote: 64")).toBeInTheDocument();
  });

  it("warns when a quote was not found in the paper", () => {
    render(wrap(<PaperPanel kind="limitations" detail={detail} ctx={ctx} />));
    expect(screen.getByText(/Quote not found \(40% match\)/)).toBeInTheDocument();
  });

  it("says plainly when something is not in the paper", () => {
    render(wrap(<PaperPanel kind="methodology" detail={detail} ctx={ctx} />));
    expect(screen.getAllByText("Not stated in the paper.")).toHaveLength(2);
    expect(screen.getByText(/no section the agent recognised as its methods/)).toBeInTheDocument();
  });
});

describe("keeping and dropping papers", () => {
  it("sends the marked papers with an optional comment, bound to the checkpoint", async () => {
    const fetch = vi.fn().mockResolvedValue(json({ action: { id: "b".repeat(32) }, kick: { submitted: false, detail: "ok" } }, 202));
    vi.stubGlobal("fetch", fetch);
    const sent = vi.fn();
    const titles = new Map([["doi:1", "Kept paper"], ["doi:2", "Dropped paper"]]);
    render(wrap(<ChoicesBar run={run} gate={gate as OpenGate} choices={{ "doi:1": "include", "doi:2": "exclude" }} titles={titles} onSent={sent} />));
    expect(screen.getByText("You marked 1 to keep and 1 to drop.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Send to the literature agent/ }));
    expect(screen.getByText("Dropped paper")).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText(/Anything else/), "More on older adults");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(sent).toHaveBeenCalled());
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({
      kind: "reject",
      stage: "literature",
      gate: "literature_review",
      payload: {
        feedback: "More on older adults",
        include: ["doi:1"],
        exclude: ["doi:2"],
        expect: { gate_nonce: "n1", artifacts: { corpus: 1 } },
      },
    });
  });

  it("labels the AI digest as such and flags numbers it may have made up", () => {
    const withDigest = {
      paper: { id: "doi:1", read: "full_text", links: {}, authors: ["A. Morgan"], title: "T", year: 2019 },
      problem: "",
      approach: "",
      methods: [],
      datasets: [],
      metrics: [],
      claims: [],
      methods_section: "",
      section_titles: [],
      gaps: [],
      open_problems: [],
      coverage: [],
      cites: [],
      cited_by: [],
      bibtex: "",
      digest: { summary: "A short summary.", design: "Crossover", sample: "24 adults", procedure: "Two nights.", gaps: ["Only young adults"], unverified_numbers: ["…in 73 percent…"] },
    } as unknown as PaperDetail;
    const ctx = { project: "sleep", runId: "r1", stage: "literature" };
    render(wrap(<PaperPanel kind="summary" detail={withDigest} ctx={ctx} />));
    expect(screen.getByText("A short summary.")).toBeInTheDocument();
    expect(screen.getByText(/AI-written · not checked against quotes/)).toBeInTheDocument();
    expect(screen.getByText(/numbers not found in the paper's text/)).toBeInTheDocument();
  });
});

describe("ScopeCard", () => {
  const view = {
    scope: {
      interpretation: "LLM agents that carry out research.",
      concepts: ["LLM-based agents", "automating research"],
      out_of_scope: ["multi-agent reinforcement learning"],
      other_readings: ["agent-based simulation of research communities"],
    },
    subtopics: [
      { name: "idea generation", paper_ids: ["doi:1", "doi:2"] },
      { name: "paper writing", paper_ids: [] },
    ],
    quality: [
      { check: "screening", value: "500 of 507 (99%) candidates passed; lowest score 6", ok: false, note: "Screening kept almost everything" },
      { check: "full text", value: "71 of 100 (71%) papers read in full", ok: true, note: "" },
    ],
  } as unknown as LiteratureView;

  it("shows how the topic was read and what to look at", async () => {
    const onFilter = vi.fn();
    render(wrap(<ScopeCard view={view} onFilter={onFilter} />));
    expect(screen.getByRole("heading", { name: "How the topic was read" })).toBeInTheDocument();
    expect(screen.getByText("LLM-based agents")).toBeInTheDocument();
    expect(screen.getByText("multi-agent reinforcement learning")).toBeInTheDocument();
    expect(screen.getByText("agent-based simulation of research communities")).toBeInTheDocument();
    expect(screen.getByText("(1 to look at)")).toBeInTheDocument();
    expect(screen.getByText("Screening kept almost everything.")).toBeInTheDocument();
    expect(screen.queryByText(/^\.$/)).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: /idea generation/ }));
    expect(onFilter).toHaveBeenCalledWith({ label: "Sub-topic: idea generation", ids: ["doi:1", "doi:2"] });
  });

  it("shows nothing for a corpus built before topics were read", () => {
    const { container } = render(wrap(<ScopeCard view={{ ...view, scope: null, quality: [] } as LiteratureView} onFilter={() => {}} />));
    expect(container).toBeEmptyDOMElement();
  });
});
