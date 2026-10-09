/** Server state: TanStack Query hooks, the live event stream, and following a decision until it is applied. */
import { useMutation, useQuery, useQueryClient, type QueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { toast } from "sonner";

import { api } from "./client";
import type {
  ActionRequest,
  ActionState,
  Affected,
  HypothesesView,
  ArtifactEnvelope,
  ArtifactInfo,
  Created,
  EventPage,
  KickResult,
  NewRun,
  Queued,
  RunDetail,
  RunEvent,
  LiteratureView,
  PaperDetail,
  PaperText,
  RunSummary,
  SearchHit,
  ServerSettings,
} from "./types";

const enc = encodeURIComponent;
const base = (project: string, runId: string) => `/runs/${enc(project)}/${enc(runId)}`;

export const keys = {
  runs: ["runs"] as const,
  settings: ["settings"] as const,
  run: (p: string, r: string) => ["run", p, r] as const,
  events: (p: string, r: string) => ["events", p, r] as const,
  artifacts: (p: string, r: string) => ["artifacts", p, r] as const,
  artifact: (p: string, r: string, name: string, version?: number) =>
    ["artifact", p, r, name, version ?? "head"] as const,
  literature: (p: string, r: string) => ["artifact", p, r, "literature"] as const,
  paper: (p: string, r: string, id: string) => ["artifact", p, r, "paper", id] as const,
  paperText: (p: string, r: string, id: string, quote?: string) =>
    ["artifact", p, r, "paper-text", id, quote ?? ""] as const,
  hypotheses: (p: string, r: string, version?: number) => ["artifact", p, r, "hypotheses", version ?? "head"] as const,
  impact: (p: string, r: string, stage: string) => ["run", p, r, "impact", stage] as const,
};

export function useSettings() {
  return useQuery({ queryKey: keys.settings, queryFn: () => api<ServerSettings>("/settings"), staleTime: 60_000 });
}

export function useRuns() {
  return useQuery({ queryKey: keys.runs, queryFn: () => api<RunSummary[]>("/runs"), refetchInterval: 15_000 });
}

export function useRun(project: string, runId: string) {
  return useQuery({
    queryKey: keys.run(project, runId),
    queryFn: () => api<RunDetail>(base(project, runId)),
    refetchInterval: 20_000, // the event stream triggers refreshes; this is the fallback
  });
}

export function useArtifacts(project: string, runId: string) {
  return useQuery({
    queryKey: keys.artifacts(project, runId),
    queryFn: () => api<ArtifactInfo[]>(`${base(project, runId)}/artifacts`),
  });
}

export function useArtifact(project: string, runId: string, name: string, version?: number, enabled = true) {
  const query = version === undefined ? "" : `?version=${version}`;
  return useQuery({
    queryKey: keys.artifact(project, runId, name, version),
    queryFn: () => api<ArtifactEnvelope>(`${base(project, runId)}/artifacts/${enc(name)}${query}`),
    enabled,
    staleTime: version === undefined ? 0 : Infinity, // a version never changes
  });
}

export function useEvents(project: string, runId: string) {
  return useQuery({
    queryKey: keys.events(project, runId),
    queryFn: () => api<EventPage>(`${base(project, runId)}/events?tail=200`),
    staleTime: Infinity, // kept current by the stream
  });
}

/** Follow the run's event stream: new events go to the activity feed and refresh what they change. */
export function useRunStream(project: string, runId: string): void {
  const queryClient = useQueryClient();
  useEffect(() => {
    if (typeof EventSource === "undefined") return;
    let refresh: ReturnType<typeof setTimeout> | undefined;
    const source = new EventSource(`/api${base(project, runId)}/events/stream`);
    source.addEventListener("events", (message) => {
      const batch = JSON.parse((message as MessageEvent).data) as { events: RunEvent[]; cursor: number };
      queryClient.setQueryData<EventPage>(keys.events(project, runId), (old) =>
        old ? { events: [...old.events, ...batch.events].slice(-500), cursor: batch.cursor } : old,
      );
      clearTimeout(refresh);
      refresh = setTimeout(() => {
        void queryClient.invalidateQueries({ queryKey: keys.run(project, runId) });
        void queryClient.invalidateQueries({ queryKey: keys.artifacts(project, runId) });
        void queryClient.invalidateQueries({ queryKey: ["artifact", project, runId], exact: false });
        void queryClient.invalidateQueries({ queryKey: keys.runs });
      }, 400);
    });
    return () => {
      clearTimeout(refresh);
      source.close();
    };
  }, [project, runId, queryClient]);
}

/** Poll a queued decision until an episode has applied (or refused) it, then say so. */
export async function followAction(
  queryClient: QueryClient,
  project: string,
  runId: string,
  actionId: string,
  label: string,
): Promise<void> {
  const deadline = Date.now() + 60 * 60 * 1000; // a GPU queue can be slow; stop asking after an hour
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    let state: ActionState;
    try {
      state = await api<ActionState>(`${base(project, runId)}/actions/${actionId}`);
    } catch {
      continue;
    }
    if (state.state === "queued") continue;
    void queryClient.invalidateQueries({ queryKey: keys.run(project, runId) });
    void queryClient.invalidateQueries({ queryKey: keys.runs });
    if (state.state === "applied") toast.success(`${label}: done`);
    else if (state.state === "rejected") toast.error(`${label} was refused: ${state.error ?? "unknown reason"}`);
    else if (state.state === "unknown") toast(`${label} was withdrawn`);
    else toast.error(`${label}: ${state.error ?? state.state}`);
    return;
  }
}

export function useDecide(project: string, runId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ request }: { request: ActionRequest; label: string }) =>
      api<Queued>(`${base(project, runId)}/actions`, { method: "POST", body: request }),
    onSuccess: (queued, { label }) => {
      void queryClient.invalidateQueries({ queryKey: keys.run(project, runId) });
      toast(`${label}: sent`, { description: queued.kick.detail });
      void followAction(queryClient, project, runId, queued.action.id, label);
    },
    onError: (error, { label }) => toast.error(`${label}: ${(error as Error).message}`),
  });
}

export function useUndo(project: string, runId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (actionId: string) => api<void>(`${base(project, runId)}/actions/${actionId}`, { method: "DELETE" }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: keys.run(project, runId) });
      toast("Decision withdrawn");
    },
    onError: (error) => toast.error((error as Error).message),
  });
}

export function useResume(project: string, runId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api<KickResult>(`${base(project, runId)}/resume`, { method: "POST" }),
    onSuccess: (kick) => {
      toast(kick.detail);
      void queryClient.invalidateQueries({ queryKey: keys.run(project, runId) });
    },
    onError: (error) => toast.error((error as Error).message),
  });
}

export function useStop(project: string, runId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (experiments: boolean) =>
      api<{ cancelled: string[] }>(`${base(project, runId)}/stop`, { method: "POST", body: { experiments } }),
    onSuccess: (result) => {
      toast(result.cancelled.length ? `Cancelled jobs ${result.cancelled.join(", ")}` : "Nothing was running");
      void queryClient.invalidateQueries({ queryKey: keys.run(project, runId) });
    },
    onError: (error) => toast.error((error as Error).message),
  });
}

export function useCreateRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: NewRun) => api<Created>("/runs", { method: "POST", body }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: keys.runs }),
  });
}

// ------------------------------------------------------------- literature
// (keys start with "artifact", so the event stream refreshes them when the corpus changes)
export function useLiterature(project: string, runId: string) {
  return useQuery({
    queryKey: keys.literature(project, runId),
    queryFn: () => api<LiteratureView>(`${base(project, runId)}/literature`),
  });
}

export function usePaper(project: string, runId: string, paperId: string, enabled = true) {
  return useQuery({
    queryKey: keys.paper(project, runId, paperId),
    queryFn: () => api<PaperDetail>(`${base(project, runId)}/literature/paper?id=${enc(paperId)}`),
    enabled,
  });
}

export function usePaperText(project: string, runId: string, paperId: string, quote?: string, enabled = true) {
  const extra = quote ? `&quote=${enc(quote)}` : "";
  return useQuery({
    queryKey: keys.paperText(project, runId, paperId, quote),
    queryFn: () => api<PaperText>(`${base(project, runId)}/literature/paper/text?id=${enc(paperId)}${extra}`),
    enabled,
  });
}

export function useLiteratureSearch(project: string, runId: string, query: string) {
  return useQuery({
    queryKey: ["artifact", project, runId, "search", query],
    queryFn: () => api<SearchHit[]>(`${base(project, runId)}/literature/search?q=${enc(query)}`),
    enabled: query.trim().length >= 2,
  });
}

// ------------------------------------------------------------- hypotheses
export function useHypotheses(project: string, runId: string, version?: number) {
  return useQuery({
    queryKey: keys.hypotheses(project, runId, version),
    queryFn: () => api<HypothesesView>(`${base(project, runId)}/hypotheses${version ? `?version=${version}` : ""}`),
  });
}

/** What a change to a stage's outputs would do to the later stages (shown before reopening a checkpoint). */
export function useImpact(project: string, runId: string, stage: string, enabled = true) {
  return useQuery({
    queryKey: keys.impact(project, runId, stage),
    queryFn: () => api<Affected[]>(`${base(project, runId)}/stages/${enc(stage)}/impact`),
    enabled,
  });
}
