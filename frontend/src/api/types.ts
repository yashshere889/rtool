import type { components } from "./schema";

type Schemas = components["schemas"];

export type RunSummary = Schemas["RunSummary"];
export type RunDetail = Schemas["RunDetail"];
export type StageSummary = Schemas["StageSummary"];
export type OpenGate = Schemas["OpenGate"];
export type Action = Schemas["Action"];
export type KickResult = Schemas["KickResult"];
export type ActionState = Schemas["ActionState"];
export type ActionRequest = Schemas["ActionRequest"];
export type Queued = Schemas["Queued"];
export type NewRun = Schemas["NewRun"];
export type Created = Schemas["Created"];
export type ArtifactInfo = Schemas["ArtifactInfo"];
export type ArtifactEnvelope = Schemas["ArtifactEnvelope"];
export type ServerSettings = Schemas["ServerSettings"];
export type WorkerState = Schemas["WorkerState"];
export type StageStatus = StageSummary["status"];

/** One line of a run's events.jsonl. */
export interface RunEvent {
  ts: string;
  run: string;
  episode: string | null;
  kind: string;
  stage?: string;
  data?: Record<string, unknown>;
}

export interface EventPage {
  events: RunEvent[];
  cursor: number;
}
