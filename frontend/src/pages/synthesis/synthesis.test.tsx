import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { RunDetail, StageSummary } from "@/api/types";
import { duration } from "@/lib/format";
import { resourceLink } from "@/lib/plan";

import { FetchPage } from "../fetch/FetchPage";
import { SynthesisPage } from "./SynthesisPage";

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

const stageOf = (id: string, agent: string, status = "done") => ({ id, agent, status, gate: null }) as unknown as StageSummary;
const runOf = (stages: StageSummary[]) =>
  ({ project: "sleep", run_id: "r1", stages, open_gates: [], queued_actions: [] }) as unknown as RunDetail;

const review = {
  topic: "Sound and sleep",
  sections: [
    {
      title: "Closed-loop stimulation",
      sentences: [{ text: "Phase-locked sound improved recall.", claims: ["C1"], papers: ["osei2021slow"] }],
    },
  ],
  dropped: [{ text: "Stimulation also cures insomnia.", claims: ["C999"], reason: "cites findings that were not offered: C999" }],
  claims: { C1: { paper_id: "p1", bib_key: "osei2021slow", text: "Recall improved", quote: "recall rose by a third" } },
  checks: [
    { name: "sentences resting on verified findings", ok: true, detail: "" },
    { name: "unsupported sentences", ok: false, detail: "1 dropped" },
  ],
  markdown_path: "work/synthesis/review.md",
};

describe("SynthesisPage", () => {
  it("shows each sentence with its citations, quotes and the sentences that were dropped", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => (url.includes("/artifacts/synthesis") ? json({ version: 1, payload: review }) : json({ events: [], cursor: 0 }))),
    );
    const stage = stageOf("synthesis", "synthesis");
    render(wrap(<SynthesisPage run={runOf([stageOf("literature", "literature"), stage])} stage={stage} />));

    await screen.findByText("Phase-locked sound improved recall.");
    expect(screen.getByText("1 sentence kept")).toBeInTheDocument();
    expect(screen.getByText("1 sentence dropped")).toBeInTheDocument();
    expect(screen.getByText("Stimulation also cures insomnia.")).toBeInTheDocument();
    expect(screen.getByText(/cites findings that were not offered/)).toBeInTheDocument();
    await userEvent.click(screen.getByText("Rests on C1"));
    expect(screen.getByText("“recall rose by a third”")).toBeInTheDocument();
    const link = screen.getByRole("link", { name: "Show in the paper" });
    expect(link.getAttribute("href")).toContain("/literature/reader?paper=p1");
    expect(screen.getByText("unsupported sentences").closest("li")).toHaveTextContent("1 dropped");
  });
});

describe("FetchPage", () => {
  it("lists the resources the plan needs, with the folder of one that has to be provided by hand", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        url.endsWith("/downloads")
          ? json({
              stage: "fetch",
              status: "awaiting_gate",
              started: true,
              last_error: null,
              plan_version: 1,
              root: "resources",
              items: [
                { name: "Cohort data", kind: "dataset", source: "manual", identifier: "cohort", status: "manual", bytes: 0, files: 0, path: "resources/cohort", note: "not provided: its folder resources/cohort is empty" },
              ],
            })
          : json({ events: [], cursor: 0 }),
      ),
    );
    const stage = stageOf("fetch", "fetch_resources", "awaiting_gate");
    render(wrap(<FetchPage run={runOf([stage])} stage={stage} />));
    await screen.findByText("Cohort data");
    expect(screen.getByText(/not provided: its folder resources\/cohort is empty/)).toBeInTheDocument();
    expect(screen.getByText("Needs you: get it by hand")).toBeInTheDocument();
  });
});

describe("FetchPage uploads", () => {
  it("adds a small file to a resource the checkpoint waits for, and says when it is in", async () => {
    const fetch = vi.fn(async (url: string, init?: RequestInit) => {
      if (url.includes("/resources/") && init?.method === "PUT") return json({ name: "Cohort data", filename: "cohort.csv", bytes: 12, files: 1, total_bytes: 12 }, 201);
      if (url.endsWith("/downloads"))
        return json({
          stage: "fetch", status: "awaiting_gate", started: true, last_error: null, plan_version: 1, root: "resources", can_upload: true,
          items: [{ name: "Cohort data", kind: "dataset", source: "manual", identifier: "cohort", status: "manual", bytes: 0, files: 0, path: "resources/cohort", note: "not provided" }],
        });
      return json({ events: [], cursor: 0 });
    });
    vi.stubGlobal("fetch", fetch);
    const stage = stageOf("fetch", "fetch_resources", "awaiting_gate");
    render(wrap(<FetchPage run={runOf([stage])} stage={stage} />));
    await userEvent.upload(await screen.findByLabelText("Add a file to Cohort data"), new File(["a,b\n1,2\n"], "cohort.csv", { type: "text/csv" }));
    await userEvent.click(screen.getByRole("button", { name: /Upload/ }));
    expect(await screen.findByText(/Added cohort.csv \(1 files there now\)/)).toBeInTheDocument();
    const [url, init] = fetch.mock.calls.find(([u, i]) => String(u).includes("/resources/") && (i as RequestInit)?.method === "PUT")!;
    expect(String(url)).toContain("/resources/Cohort%20data/inbox?filename=cohort.csv");
    expect((init as RequestInit).headers).toMatchObject({ "X-Requested-With": "rtool" });
  });
});

describe("helpers", () => {
  it("link the newer resource sources and say durations in words", () => {
    expect(resourceLink({ source: "geo", kind: "dataset", identifier: "GSE12345" })).toBe("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE12345");
    expect(resourceLink({ source: "pdb", kind: "other", identifier: "1abc" })).toBe("https://www.rcsb.org/structure/1abc");
    expect(resourceLink({ source: "local", kind: "dataset", identifier: "mine" })).toBeNull();
    expect([duration(8), duration(90), duration(4500), duration(7200)]).toEqual(["8 s", "1 min 30 s", "1 h 15 min", "2 h"]);
  });
});
