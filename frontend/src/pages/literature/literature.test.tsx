import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { PaperDetail, RunDetail } from "@/api/types";

import { PaperPanel } from "./PaperPanels";
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
    expect(screen.getByText(/Quote found in the paper \(100% match\)/)).toBeInTheDocument();
    expect(screen.getByText("recall = 2.3")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Show in paper/ })).toHaveAttribute(
      "href",
      "/runs/sleep/r1/literature/reader?paper=doi%3A1&claim=0",
    );
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
