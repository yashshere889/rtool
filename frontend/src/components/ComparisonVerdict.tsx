import type { ExperimentResults } from "@/api/types";

import { Badge } from "./ui/badge";

export type Comparison = ExperimentResults["comparisons"][number];

/** Why a comparison has no p-value (mirrors Comparison.verdict on the server). */
export function untested(c: Comparison): string {
  if (c.n < 2) return "Too few runs to test.";
  if (c.constant_difference) return "The difference was identical for every seed: it may be a fixed offset, not a measured effect.";
  return "The results did not vary across seeds, so there is nothing to test against.";
}

/** One condition against the baseline: better, worse, no clear difference, or not testable. */
export function ComparisonVerdict({ c }: { c: Comparison }) {
  if (c.p_value === null || c.p_value === undefined)
    return (
      <span title={untested(c)}>
        <Badge tone="warn">No test possible</Badge>
      </span>
    );
  if (!c.significant) return <Badge>No clear difference</Badge>;
  return c.better ? <Badge tone="ok">Better than baseline</Badge> : <Badge tone="bad">Worse than baseline</Badge>;
}
