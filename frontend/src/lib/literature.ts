/** Words and small helpers for papers. */
import type { PaperCardData } from "@/api/types";

import type { Tone } from "./stages";

export function authorsShort(authors: string[], max = 3): string {
  if (!authors.length) return "Unknown authors";
  return authors.length > max ? `${authors.slice(0, max).join(", ")} et al.` : authors.join(", ");
}

export function paperStatus(paper: PaperCardData): { label: string; tone: Tone } {
  if (!paper.included) return { label: "Excluded", tone: "neutral" };
  if (paper.needs_upload) return { label: "No open copy — upload a PDF", tone: "warn" };
  if (paper.status === "failed") return { label: "Could not be read", tone: "bad" };
  if (paper.read === "full_text") return { label: "Read in full", tone: "ok" };
  if (paper.read === "abstract") return { label: "Abstract only", tone: "info" };
  return { label: "Not read yet", tone: "neutral" };
}

export const GAP_KINDS: Record<string, string> = {
  uncovered_subtopic: "Sub-topic with few papers",
  untested_combination: "Untested combination",
  thin_topic: "Few papers on this topic",
  stale_topic: "No recent work",
  unresolved_limitation: "Unresolved limitation",
};

/** "Morgan, A., Chen, L. (2019). Title. Venue." — enough to paste into notes. */
export function plainCitation(paper: PaperCardData): string {
  const authors = authorsShort(paper.authors, 6);
  const venue = paper.venue ? ` ${paper.venue}.` : "";
  const link = paper.links.doi ?? paper.links.arxiv ?? "";
  return `${authors} (${paper.year ?? "n.d."}). ${paper.title}.${venue}${link ? ` ${link}` : ""}`;
}

export function foundBy(paper: PaperCardData): string {
  const first = paper.found_by[0] ?? "";
  if (first.startsWith("query:")) return `found by searching “${first.slice(6)}”`;
  if (first.startsWith("refs:")) return "found in the reference list of another paper";
  if (first.startsWith("cites:")) return "found because it cites another paper";
  if (first === "manual") return "added by you";
  return "";
}
