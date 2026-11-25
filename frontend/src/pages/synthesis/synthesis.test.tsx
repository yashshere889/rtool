import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
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

const viewOf = (synthesis: typeof review, extra: object = {}) => ({
  stage: "synthesis", version: 1, versions: [1], head: 1, synthesis, by: null, edited_from: null, edit_note: "",
  created_at: new Date().toISOString(), editable: true, why_not_editable: "", ...extra,
});

describe("SynthesisPage", () => {
  it("shows each sentence with its citations, quotes and the sentences that were dropped", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => (url.endsWith("/synthesis") ? json(viewOf(review)) : json({ events: [], cursor: 0 }))),
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

describe("ReviewEditor", () => {
  it("checks a rewritten sentence as it is typed, refuses one that invents a number, and sends the edit", async () => {
    const fetch = vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      if (url.endsWith("/synthesis") && method === "GET") return json(viewOf(review));
      if (url.endsWith("/synthesis/check")) {
        const text = JSON.parse(String(init?.body)).value.sections[0].sentences[0].text as string;
        return text.includes("97")
          ? json({ ok: false, error: "sentence 1 of “Closed-loop stimulation”: states numbers its findings do not: 97", synthesis: null, changed: [] })
          : json({ ok: true, error: "", synthesis: review, changed: ["sections"] });
      }
      if (url.endsWith("/actions")) return json({ action: { id: "a".repeat(32) }, kick: { submitted: false, detail: "ok" } }, 202);
      return json({ events: [], cursor: 0 });
    });
    vi.stubGlobal("fetch", fetch);
    const stage = stageOf("synthesis", "synthesis", "awaiting_gate");
    render(wrap(<SynthesisPage run={runOf([stageOf("literature", "literature"), stage])} stage={stage} />));
    await userEvent.click(await screen.findByRole("button", { name: /Edit the review/ }));
    const sentence = await screen.findByRole("textbox", { name: "Sentence 1 of Closed-loop stimulation" });
    await userEvent.clear(sentence);
    await userEvent.type(sentence, "It helped 97 % of people.");
    expect(await screen.findByText(/states numbers its findings do not: 97/, {}, { timeout: 3000 })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Apply my edits" })).toBeDisabled();
    await userEvent.clear(sentence);
    await userEvent.type(sentence, "Phase-locked sound improved recall in the study.");
    expect(await screen.findByText("Every sentence rests on verified findings.", {}, { timeout: 3000 })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: "Apply my edits" })).toBeEnabled(), { timeout: 3000 });
    await userEvent.click(screen.getByRole("button", { name: "Apply my edits" }));
    await waitFor(() => expect(fetch.mock.calls.some(([u]) => String(u).endsWith("/actions"))).toBe(true));
    const [, init] = fetch.mock.calls.find(([u]) => String(u).endsWith("/actions"))!;
    const sent = JSON.parse(String((init as RequestInit).body));
    expect(sent).toMatchObject({ kind: "edit", stage: "synthesis", payload: { artifact: "synthesis", base_version: 1 } });
    expect(sent.payload.value.sections[0].sentences[0]).toMatchObject({ text: "Phase-locked sound improved recall in the study.", claims: ["C1"] });
  });

  it("offers no editing when the review cannot be edited now", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => (url.endsWith("/synthesis") ? json(viewOf(review, { editable: false, why_not_editable: "stage is running" })) : json({ events: [], cursor: 0 }))));
    const stage = stageOf("synthesis", "synthesis");
    render(wrap(<SynthesisPage run={runOf([stageOf("literature", "literature"), stage])} stage={stage} />));
    await screen.findByText("Phase-locked sound improved recall.");
    expect(screen.queryByRole("button", { name: /Edit the review/ })).toBeNull();
  });
});
