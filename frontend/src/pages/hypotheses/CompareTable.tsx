import type { ReactNode } from "react";

import type { HypothesisCardData } from "@/api/types";
import { cn } from "@/lib/cn";
import { COMPUTE, CRITERIA, score100, verifiedEvidence } from "@/lib/hypotheses";

interface Row {
  label: string;
  value: (card: HypothesisCardData) => ReactNode;
  best?: (card: HypothesisCardData) => number; // higher is better; the best cell is marked
}

const ROWS: Row[] = [
  { label: "Overall score (of 100)", value: (c) => score100(c.hypothesis), best: (c) => c.hypothesis.total_score },
  ...CRITERIA.map((criterion) => ({
    label: `${criterion.label} (of 5)`,
    value: (c: HypothesisCardData) => c.hypothesis.scores[criterion.key] ?? "–",
    best: (c: HypothesisCardData) => c.hypothesis.scores[criterion.key] ?? 0,
  })),
  {
    label: "Novelty check",
    value: (c) => {
      const n = c.hypothesis.novelty;
      return n ? `${n.novel ? "New" : "May repeat earlier work"} (${n.score}/10)` : "Not checked";
    },
    best: (c) => c.hypothesis.novelty?.score ?? 0,
  },
  {
    label: "Citations found in the papers",
    value: (c) => `${verifiedEvidence(c.hypothesis)} of ${c.hypothesis.evidence.length}`,
  },
  { label: "Papers used", value: (c) => c.papers.length },
  {
    label: "Compute",
    value: (c) => `${COMPUTE[c.hypothesis.compute].label} (${COMPUTE[c.hypothesis.compute].detail})`,
    best: (c) => -c.gpu_hours,
  },
  { label: "Measured", value: (c) => c.hypothesis.dependent_variables.join("; ") },
  { label: "Data", value: (c) => c.hypothesis.datasets.join("; ") || "None named" },
  { label: "Methods", value: (c) => c.hypothesis.methods.join("; ") || "None named" },
  { label: "Risks", value: (c) => [...c.hypothesis.feasibility_notes, ...c.hypothesis.risks].join("; ") || "None noted" },
  { label: "Judges", value: (c) => c.hypothesis.rubric_comments.join(" ") || "–" },
];

/** The shortlist as a table: one column per hypothesis, the best value in each scored row marked. */
export function CompareTable({ cards, chosen }: { cards: HypothesisCardData[]; chosen: number | null }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-line bg-surface">
      <table className="w-full min-w-[40rem] text-sm">
        <caption className="sr-only">The hypotheses compared</caption>
        <thead>
          <tr className="border-b border-line">
            <th scope="col" className="w-48 p-3 text-left text-xs font-semibold tracking-wide text-ink-soft uppercase" />
            {cards.map((card) => (
              <th key={card.hypothesis.id} scope="col" className="p-3 text-left align-bottom font-semibold">
                <span className="block text-xs font-medium text-ink-soft">
                  Option {(card.index ?? 0) + 1}
                  {card.index === chosen ? " · your choice" : ""}
                </span>
                {card.hypothesis.title}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ROWS.map((row) => {
            const scores = row.best ? cards.map(row.best) : [];
            const top = scores.length ? Math.max(...scores) : null;
            const unique = top !== null && scores.filter((s) => s === top).length < cards.length;
            return (
              <tr key={row.label} className="border-b border-line last:border-0">
                <th scope="row" className="p-3 text-left align-top font-medium text-ink-soft">
                  {row.label}
                </th>
                {cards.map((card, i) => (
                  <td
                    key={card.hypothesis.id}
                    className={cn("p-3 align-top", unique && scores[i] === top && "bg-ok-soft/60 font-semibold text-ok")}
                  >
                    {row.value(card)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
