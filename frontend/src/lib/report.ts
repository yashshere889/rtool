/** Reports in everyday words: who wrote each section, and the review's criteria. */
import type { Tone } from "./stages";

export const ORIGINS: Record<string, { label: string; tone: Tone; help: string }> = {
  rtool: { label: "Facts from the results", tone: "neutral", help: "Written by rtool from the plan and results; not editable." },
  llm: { label: "Drafted by the writer", tone: "info", help: "Drafted by the AI writer under rtool's checks." },
  reviewer: { label: "Edited by the reviewer", tone: "warn", help: "Rewritten by the AI reviewer to address review points." },
  human: { label: "Written by you", tone: "ok", help: "Your text: kept when the writer runs again." },
};

export const CRITERIA: Record<string, { label: string; help: string }> = {
  soundness: { label: "Soundness", help: "Do the conclusions follow from the results and statistics?" },
  evidence: { label: "Evidence", help: "Are claims about prior work supported by the cited papers?" },
  clarity: { label: "Clarity", help: "Is it clear and well organised?" },
  reproducibility: { label: "Reproducibility", help: "Could someone rerun it from the report and the code?" },
  limitations: { label: "Limitations", help: "Are limitations and threats to validity discussed honestly?" },
};

/** "Abstract", then "1. Introduction", "2. Related work", ... as in the rendered document. */
export function sectionHeadings(sections: { id: string; title: string }[]): Record<string, string> {
  let n = 0;
  return Object.fromEntries(sections.map((s) => [s.id, s.id === "abstract" ? s.title : `${++n}. ${s.title}`]));
}

export interface ReportDraft {
  title: string;
  templates: Record<string, string>;
}
