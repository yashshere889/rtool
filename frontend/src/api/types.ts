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

export type LiteratureView = Schemas["LiteratureView"];
export type PaperCardData = Schemas["PaperCard"];
export type PaperDetail = Schemas["PaperDetail"];
export type PaperText = Schemas["PaperText"];
export type ClaimView = Schemas["ClaimView"];
export type PaperRef = Schemas["PaperRef"];
export type GapView = Schemas["GapView"];
export type OpenProblemView = Schemas["OpenProblemView"];
export type CoverageView = Schemas["CoverageView"];
export type SearchHit = Schemas["SearchHit"];
export type Uploaded = Schemas["Uploaded"];

export type HypothesesView = Schemas["HypothesesView"];
export type HypothesisCardData = Schemas["HypothesisCard"];
export type Hypothesis = Schemas["Hypothesis"];
export type PaperUse = Schemas["PaperUse"];
export type Selection = Schemas["Selection"];
export type Affected = Schemas["Affected"];

export type PlanView = Schemas["PlanView"];
export type ExperimentPlan = Schemas["ExperimentPlan"];
export type PlanCheck = Schemas["PlanCheck"];
export type ClusterView = Schemas["ClusterView"];
export type PartitionView = Schemas["PartitionView"];
export type DownloadsView = Schemas["DownloadsView"];
export type DownloadItem = Schemas["DownloadItem"];
export type Draft = Schemas["Draft"];
export type ResourceSpec = Schemas["ResourceSpec"];

export type CodeView = Schemas["CodeView"];
export type CodeFileView = Schemas["CodeFileView"];
export type RunCell = Schemas["RunCell"];
export type FileText = Schemas["FileText"];
export type CodeCheck = Schemas["CodeCheck"];
export type LogText = Schemas["LogText"];
export type ResultsView = Schemas["ResultsView"];
export type ExperimentResults = Schemas["ExperimentResults"];

export type ReportView = Schemas["ReportView"];
export type SectionView = Schemas["SectionView"];
export type ReportCheck = Schemas["ReportCheck"];
export type ReviewView = Schemas["ReviewView"];
export type PaperInfo = Schemas["PaperInfo"];
export type PlaceholderInfo = Schemas["PlaceholderInfo"];
export type Finding = Schemas["Finding"];
export type ReportWhich = ReportView["artifact"];
