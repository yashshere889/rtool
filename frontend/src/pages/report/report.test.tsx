import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { forwardRef, useImperativeHandle, type ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { PaperInfo, ReportView, ReviewView, RunDetail, StageSummary } from "@/api/types";
import { linkCitations, Markdown } from "@/components/Markdown";
import { sectionHeadings } from "@/lib/report";

import { ReportWorkspace } from "./ReportWorkspace";
import { ReviewPage } from "./ReviewPage";

// CodeMirror needs a real browser layout; a text box (with the same insert API) stands in for it here
vi.mock("@/components/code/CodeEditor", () => ({
  default: () => null,
  CodeDiff: () => null,
  MarkdownEditor: forwardRef(function Fake(
    { value, onChange, label }: { value: string; onChange: (v: string) => void; label: string },
    ref,
  ) {
    useImperativeHandle(ref, () => ({ insert: (text: string) => onChange(value + text) }));
    return <textarea aria-label={label} value={value} onChange={(e) => onChange(e.target.value)} />;
  }),
}));

function wrap(children: ReactNode, path = "/") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>{children}</MemoryRouter>
    </QueryClientProvider>
  );
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

afterEach(() => vi.unstubAllGlobals());

const morgan: PaperInfo = {
  key: "morgan2019",
  paper_id: "doi:1",
  title: "Phase-locked stimulation",
  authors: "Alice Morgan, Lei Chen et al.",
  year: 2019,
  label: "Morgan et al., 2019",
};

const view = {
  artifact: "report",
  stage: "report",
  version: 1,
  versions: [1],
  head: 1,
  by: null,
  edited_from: null,
  edit_note: "",
  created_at: new Date().toISOString(),
  title: "Timing matters",
  title_origin: "llm",
  sections: [
    { id: "abstract", title: "Abstract", origin: "llm", template: "We ran {{runs_ok}} runs.", rendered: "We ran 12 runs.", editable: true, problems: [] },
    { id: "methods", title: "Methods", origin: "rtool", template: "**Design.** x", rendered: "**Design.** x", editable: false, problems: [] },
    {
      id: "discussion",
      title: "Discussion",
      origin: "llm",
      template: "As shown before [@morgan2019].",
      rendered: "As shown before [@morgan2019].",
      editable: true,
      problems: [],
    },
  ],
  findings: [{ kind: "incomplete_runs", detail: "1 of 12 runs failed", section: "", severity: "minor", target: "report" }],
  citations: ["morgan2019"],
  references: [morgan],
  figures: [],
  markdown_path: "work/report/report/report.md",
  bibtex_path: "work/report/report/refs.bib",
  base_dir: "work/report/report",
  placeholders: [{ key: "mean:acc:new", text: "0.72", meaning: "mean acc of new (3 runs)" }],
  papers: [morgan],
  editable: true,
  why_not_editable: "",
} as unknown as ReportView;

const stage = { id: "report", agent: "writer", status: "done", gate: null } as unknown as StageSummary;
const run = { project: "sleep", run_id: "r1", stages: [stage], open_gates: [], queued_actions: [] } as unknown as RunDetail;

describe("report helpers", () => {
  it("number the sections and link citations to papers", () => {
    expect(sectionHeadings(view.sections)).toEqual({ abstract: "Abstract", methods: "1. Methods", discussion: "2. Discussion" });
    const papers = new Map([[morgan.key, morgan]]);
    expect(linkCitations("Known [@morgan2019; @other].", papers)).toBe("Known ([Morgan et al., 2019](cite:morgan2019); [other](cite:other)).");
    render(wrap(<Markdown text="Seen in [@morgan2019]." papers={papers} project="sleep" runId="r1" stage="report" baseDir="work/report" />));
    expect(screen.getByRole("link", { name: "Morgan et al., 2019" })).toHaveAttribute("href", "/runs/sleep/r1/report/reader?paper=doi%3A1");
  });
});

describe("ReportWorkspace", () => {
  it("reads as a document and sends checked edits", async () => {
    const fetch = vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      if (url.includes("/report?")) return json(view);
      if (url.includes("/drafts/") && method === "GET") return json({ detail: "no draft" }, 404);
      if (url.includes("/drafts/")) return json({ ...JSON.parse(String(init?.body)), saved_at: new Date().toISOString() });
      if (url.endsWith("/report/check")) {
        const value = JSON.parse(String(init?.body)).value;
        const discussion = value.sections.find((s: { id: string }) => s.id === "discussion");
        const bad = discussion.template.includes("{{nope}}");
        return json(
          bad
            ? { ok: false, error: "Discussion: unknown placeholder {{nope}}", title: "", sections: [], findings: [], changed: [] }
            : { ok: true, error: "", title: value.title, sections: view.sections, findings: [], changed: ["sections"] },
        );
      }
      if (url.endsWith("/impact")) return json([{ stage: "review", status: "awaiting_gate", effect: "redo", detail: "" }]);
      if (url.endsWith("/actions")) return json({ action: { id: "a".repeat(32) }, kick: { submitted: false, detail: "ok" } }, 202);
      return json({ state: "queued" });
    });
    vi.stubGlobal("fetch", fetch);
    render(wrap(<ReportWorkspace run={run} which="report" stage="report" />));

    const article = await screen.findByRole("article");
    expect(within(article).getByRole("heading", { name: "Timing matters" })).toBeInTheDocument();
    expect(within(article).getByRole("link", { name: "Morgan et al., 2019" })).toBeInTheDocument();
    expect(screen.getByText("1 of 12 runs failed")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Edit" }));
    expect(await screen.findByText(/written by rtool from the plan and results/i)).toBeInTheDocument(); // methods locked
    const discussion = await screen.findByRole("textbox", { name: "2. Discussion" });
    await userEvent.type(discussion, " Also {{{{nope}}");
    await screen.findByText(/unknown placeholder \{\{nope\}\}/, {}, { timeout: 3000 });
    expect(screen.getByRole("button", { name: "Apply my edits" })).toBeDisabled();

    await userEvent.clear(discussion);
    await userEvent.type(discussion, "Our own view.");
    await userEvent.click(screen.getByRole("button", { name: /Insert a result/ }));
    await userEvent.click(await screen.findByRole("button", { name: /0.72/ }));
    expect(discussion).toHaveValue("Our own view.{{mean:acc:new}}");
    expect(await screen.findByText(/Will be redone: Review/)).toBeInTheDocument();
    const apply = screen.getByRole("button", { name: "Apply my edits" });
    await waitFor(() => expect(apply).toBeEnabled(), { timeout: 3000 });
    await userEvent.click(apply);
    await waitFor(() => expect(fetch.mock.calls.some(([u]) => String(u).endsWith("/actions"))).toBe(true));
    const [, init] = fetch.mock.calls.find(([u]) => String(u).endsWith("/actions"))!;
    const sent = JSON.parse(String(init!.body));
    expect(sent).toMatchObject({ kind: "edit", stage: "report", payload: { artifact: "report", base_version: 1 } });
    expect(sent.payload.value.sections.find((s: { id: string }) => s.id === "discussion").template).toBe("Our own view.{{mean:acc:new}}");
  });
});

describe("ReviewPage", () => {
  it("shows the verdict, scores, checks and what became of each issue", async () => {
    const review = {
      stage: "review",
      version: 1,
      review: {
        verdict: "needs_human",
        summary: "Mostly sound; one claim overreaches.",
        scores: { soundness: 2.5, evidence: 4, clarity: 4, reproducibility: 5, limitations: 4 },
        checks: [
          { name: "citations exist", ok: true, detail: "3 papers cited" },
          { name: "numbers trace to the results", ok: false, detail: "discussion: number typed" },
        ],
        issues: [
          { section: "results", problem: "No test named", suggestion: "Name it", target: "report", severity: "minor" },
          { section: "discussion", problem: "Overclaims", suggestion: "Soften", target: "report", severity: "major" },
        ],
        edited_sections: ["results"],
        change_requests: [],
        review_path: "",
        diff_path: "",
      },
      final_version: 1,
      final_edited: false,
      diff: "--- report.md\n+++ final_report.md\n@@ -1 +1 @@\n-old\n+new\n",
      min_score: 3,
      rounds_used: 1,
      rounds_max: 2,
      report_stage: "report",
    } as unknown as ReviewView;
    vi.stubGlobal("fetch", vi.fn(async (url: string) => (url.endsWith("/review") ? json(review) : json({ detail: "no" }, 404))));
    const reviewStage = { id: "review", agent: "reviewer", status: "awaiting_gate", gate: null } as unknown as StageSummary;
    render(wrap(<ReviewPage run={{ ...run, stages: [reviewStage] } as unknown as RunDetail} stage={reviewStage} />));

    expect(await screen.findByText("The report needs your judgment")).toBeInTheDocument();
    expect(screen.getByText("1 of 2 passed", { exact: false })).toBeInTheDocument();
    expect(screen.getByLabelText("2.5 out of 5")).toBeInTheDocument();
    expect(screen.getByText("Edited by the reviewer")).toBeInTheDocument();
    expect(screen.getByText("Open: yours to decide")).toBeInTheDocument();
    expect(screen.getByText(/1 of at most 2 revision rounds/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("tab", { name: /Reviewer's edits/ }));
    expect(await screen.findByText("+new")).toBeInTheDocument();
  });
});
