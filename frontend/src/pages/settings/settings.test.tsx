import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import { NewResearch } from "../NewResearch";
import { Settings } from "../Settings";

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

const packs = [
  { name: "generic", title: "Any field", pipeline: "research", description: "Any computational research.", max_gpu_hours: 12 },
  { name: "medicine", title: "Medicine and public health", pipeline: "synthesis", description: "Literature reviews.", max_gpu_hours: null },
];
const settings = { version: "0.1.0", mode: "job", runs_locally: false, runs_root: "/runs", secrets: {}, packs };

const fetchFor = (extra: (url: string) => unknown) =>
  vi.fn(async (url: string) => {
    if (url.endsWith("/settings")) return json(settings);
    if (url.endsWith("/fields"))
      return json([
        { name: "generic", title: "Any field", description: "Any computational research.", pipeline: "research", origin: "built-in", state: "offered", detail: "", max_gpu_hours: 12 },
        { name: "omics", title: "Omics", description: "", pipeline: "research", origin: "built-in", state: "not_offered", detail: "", max_gpu_hours: null },
        { name: "dup", title: "", description: "", pipeline: "", origin: "acme 1.0", state: "shadowed", detail: "another pack has this name, so this one is never used", max_gpu_hours: null },
      ]);
    if (url.endsWith("/pipelines"))
      return json([
        { name: "study", description: "A study.", stages: [{ id: "protocol", agent: "protocol", after: [], gate: "protocol_review", produces: ["protocol"] }] },
      ]);
    if (url.endsWith("/agents"))
      return json([{ name: "protocol", description: "Designs a study.", consumes: ["research_brief"], optional: [], produces: ["protocol"], uses_llm: true }]);
    return json(extra(url));
  });

describe("Settings", () => {
  it("lists the fields with what each runs and whether it is offered", async () => {
    vi.stubGlobal("fetch", fetchFor(() => []));
    render(wrap(<Settings />));
    await userEvent.click(await screen.findByRole("tab", { name: "Fields" }));
    await screen.findByText("Any field");
    expect(screen.getByText("at most 12 GPU-hours")).toBeInTheDocument();
    expect(screen.getByText("Not offered here")).toBeInTheDocument();
    expect(screen.getByText("Never used")).toBeInTheDocument();
    expect(screen.getByText(/another pack has this name/)).toBeInTheDocument();
  });

  it("shows each pipeline's stages and checkpoints, and the agents", async () => {
    vi.stubGlobal("fetch", fetchFor(() => []));
    render(wrap(<Settings />));
    await userEvent.click(await screen.findByRole("tab", { name: "Pipelines & agents" }));
    await screen.findByText("Study pipeline");
    expect(screen.getByText("checkpoint: Protocol review")).toBeInTheDocument();
    expect(screen.getByText("uses the model")).toBeInTheDocument();
    expect(screen.getByText(/writes protocol/)).toBeInTheDocument();
  });

  it("runs the machine checks only when asked, and contacts the paper APIs only on request", async () => {
    const fetch = fetchFor(() => [
      { name: "rtool", status: "ok", detail: "0.1.0 on Python 3.12" },
      { name: "slurm", status: "skip", detail: "slurm.backend is 'fake'" },
    ]);
    vi.stubGlobal("fetch", fetch);
    render(wrap(<Settings />));
    await userEvent.click(await screen.findByRole("tab", { name: "Checks" }));
    expect(screen.getByText(/Nothing has been checked yet/)).toBeInTheDocument();
    expect(fetch.mock.calls.some(([u]) => String(u).includes("/doctor"))).toBe(false);
    await userEvent.click(screen.getByRole("button", { name: "Run the checks" }));
    await screen.findByText("0.1.0 on Python 3.12");
    expect(fetch.mock.calls.some(([u]) => String(u).endsWith("/doctor?network=false"))).toBe(true);
    expect(fetch.mock.calls.some(([u]) => String(u).includes("network=true"))).toBe(false);
    await userEvent.click(screen.getByRole("button", { name: "Also reach the paper APIs" }));
    await waitFor(() => expect(fetch.mock.calls.some(([u]) => String(u).endsWith("/doctor?network=true"))).toBe(true));
  });
});

describe("NewResearch", () => {
  it("says what the chosen field runs and the cap this server puts on it", async () => {
    vi.stubGlobal("fetch", fetchFor(() => ({})));
    render(wrap(<NewResearch />));
    await userEvent.click(await screen.findByRole("button", { name: /More options/ }));
    expect(await screen.findByText(/An experiment on the cluster. Any computational research./)).toBeInTheDocument();
    expect(screen.getByText("This server caps Any field at 12 GPU-hours.")).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText("Field"), "medicine");
    expect(screen.getByText(/A literature review only, with no experiment. Literature reviews./)).toBeInTheDocument();
    expect(screen.getByText("Optional cap for the whole study.")).toBeInTheDocument();
  });
});
