/** Hypotheses in everyday words: rubric criteria, compute classes, why a paper is listed, and what can be edited. */
import type { Hypothesis, PaperUse } from "@/api/types";

import { humanize } from "./format";

export const CRITERIA: { key: string; label: string; help: string }[] = [
  { key: "novelty", label: "Novelty", help: "Is it new compared with the papers that were read?" },
  { key: "feasibility", label: "Feasibility", help: "Can it be done with the data and compute available?" },
  { key: "testability", label: "Testability", help: "Does it make a clear, measurable prediction?" },
  { key: "significance", label: "Significance", help: "Would the answer matter to the field?" },
  { key: "clarity", label: "Clarity", help: "Is it stated precisely enough to plan an experiment?" },
];

export const COMPUTE: Record<Hypothesis["compute"], { label: string; detail: string }> = {
  cpu_only: { label: "CPU only", detail: "needs no GPU time" },
  small_gpu: { label: "Small GPU job", detail: "up to 8 GPU-hours" },
  medium_gpu: { label: "Medium GPU job", detail: "8 to 72 GPU-hours" },
  large_gpu: { label: "Large GPU job", detail: "more than 72 GPU-hours" },
};

export const ROLES: Record<PaperUse["roles"][number], { label: string; help: string }> = {
  evidence: { label: "Quoted as evidence", help: "A sentence from this paper supports the hypothesis." },
  closest: { label: "Closest earlier work", help: "The novelty check found this paper most similar." },
  related: { label: "Behind the gap", help: "The gap or open problem this hypothesis addresses comes from it." },
};

export type FieldKind = "line" | "text" | "list" | "compute";

export interface EditableField {
  key: keyof Hypothesis & string;
  label: string;
  kind: FieldKind;
  hint?: string;
}

/** What a researcher may change before choosing (evidence, scores and checks stay as the agent found them). */
export const EDITABLE: EditableField[] = [
  { key: "title", label: "Title", kind: "line" },
  { key: "statement", label: "Hypothesis", kind: "text", hint: "If or when <condition>, then <measurable outcome>." },
  { key: "null_hypothesis", label: "Null hypothesis", kind: "text" },
  { key: "rationale", label: "Why it follows from the literature", kind: "text" },
  { key: "independent_variables", label: "Independent variables (what is varied)", kind: "list", hint: "One per line." },
  { key: "dependent_variables", label: "Dependent variables (what is measured)", kind: "list", hint: "One per line." },
  { key: "controls", label: "Controls", kind: "list", hint: "One per line." },
  { key: "datasets", label: "Datasets, populations or resources", kind: "list", hint: "One per line." },
  { key: "methods", label: "Methods", kind: "list", hint: "One per line." },
  {
    key: "success_criteria",
    label: "Success criteria",
    kind: "text",
    hint: "What result would support it, and what would refute it.",
  },
  { key: "risks", label: "Risks", kind: "list", hint: "One per line." },
  { key: "compute", label: "Compute it needs", kind: "compute" },
];

export function fieldLabel(key: string): string {
  return EDITABLE.find((f) => f.key === key)?.label ?? humanize(key);
}

export type Draft = Record<string, string>;

/** The editable fields as form text (lists one item per line). */
export function toDraft(h: Hypothesis): Draft {
  const draft: Draft = {};
  for (const field of EDITABLE) {
    const value = h[field.key];
    draft[field.key] = Array.isArray(value) ? value.join("\n") : String(value ?? "");
  }
  return draft;
}

function parse(field: EditableField, text: string): string | string[] {
  if (field.kind === "list") return text.split("\n").map((line) => line.trim()).filter(Boolean);
  return text.trim();
}

/** Only the fields the researcher changed, in the shape the hypothesis has. */
export function changedFields(h: Hypothesis, draft: Draft): Record<string, string | string[]> {
  const edit: Record<string, string | string[]> = {};
  for (const field of EDITABLE) {
    const next = parse(field, draft[field.key] ?? "");
    if (JSON.stringify(next) !== JSON.stringify(h[field.key])) edit[field.key] = next;
  }
  return edit;
}

/** Problems that would make the edited hypothesis unusable for planning. */
export function draftProblems(draft: Draft): string[] {
  const problems: string[] = [];
  for (const key of ["title", "statement", "null_hypothesis", "success_criteria"] as const) {
    if (!draft[key]?.trim()) problems.push(`${fieldLabel(key)} cannot be empty.`);
  }
  if (!draft.independent_variables?.trim()) problems.push("Name at least one independent variable.");
  if (!draft.dependent_variables?.trim()) problems.push("Name at least one dependent variable.");
  return problems;
}

export function verifiedEvidence(h: Hypothesis): number {
  return h.evidence.filter((e) => e.verified).length;
}

/** The total score (0 to 1) as a whole number out of 100. */
export function score100(h: Hypothesis): number {
  return Math.round(h.total_score * 100);
}
