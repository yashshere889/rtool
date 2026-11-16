import { AlertTriangle, CheckCircle2, Download } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router";

import { runPath } from "@/api/client";
import { useHypotheses, useReport, useResults, useReview } from "@/api/hooks";
import type { RunDetail } from "@/api/types";
import { num, pValue } from "@/lib/code";

import { ComparisonVerdict } from "./ComparisonVerdict";
import { Badge } from "./ui/badge";
import { buttonStyles } from "./ui/button";

const OUTCOME = {
  supported: { label: "Supported", tone: "ok" },
  refuted: { label: "Refuted", tone: "bad" },
  inconclusive: { label: "Inconclusive", tone: "warn" },
} as const;

const linkStyle = "text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent";

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid gap-1 sm:grid-cols-[9rem_minmax(0,1fr)] sm:gap-4">
      <dt className="text-sm font-medium text-ink-soft">{label}</dt>
      <dd className="min-w-0 text-sm">{children}</dd>
    </div>
  );
}

/** A finished run at a glance: what was tested, the primary outcome, the reviewer's verdict, and the downloads.
 * Every sentence is built from the run's own artifacts; a part whose stage this pipeline lacks is left out. */
export function FinishedSummary({ run }: { run: RunDetail }) {
  const hypotheses = useHypotheses(run.project, run.run_id);
  const results = useResults(run.project, run.run_id);
  const review = useReview(run.project, run.run_id);
  const report = useReport(run.project, run.run_id, "report"); // the Writer applied the success criterion
  const base = runPath(run.project, run.run_id);
  const stageOf = (agent: string) => run.stages.find((s) => s.agent === agent)?.id;

  const selection = hypotheses.data?.selection;
  const r = results.data;
  const primary = r?.metrics.find((m) => m.primary);
  const comparisons = primary ? (r?.results.comparisons ?? []).filter((c) => c.metric === primary.name) : [];
  const accepted = review.data?.review.verdict === "accept";
  const outcome = report.data?.hypothesis_outcome ? OUTCOME[report.data.hypothesis_outcome] : null;

  return (
    <section aria-labelledby="finished-heading" className="rounded-xl border border-ok/30 bg-ok-soft/30 p-5">
      <p className="flex items-center gap-2 text-xs font-semibold tracking-wide text-ok uppercase">
        <CheckCircle2 className="size-4" /> Finished
      </p>
      <h2 id="finished-heading" className="mt-1 text-lg font-semibold">
        What this study found
      </h2>

      <dl className="mt-4 space-y-3">
        {selection && (
          <Row label="Hypothesis tested">
            {stageOf("hypothesis") ? (
              <Link to={`${base}/${stageOf("hypothesis")}`} className={linkStyle}>
                {selection.title}
              </Link>
            ) : (
              selection.title
            )}
            {selection.edited && <span className="text-ink-soft"> (edited by you)</span>}
            {outcome && (
              <div className="mt-2">
                <p className="flex flex-wrap items-center gap-2">
                  <Badge tone={outcome.tone}>{outcome.label}</Badge>
                  <span className="text-ink-soft">judged against its success criterion</span>
                </p>
                {report.data!.hypothesis_checks.length > 0 && (
                  <ul aria-label="How the criterion was applied" className="mt-1.5 list-disc space-y-0.5 pl-5 text-ink-soft">
                    {report.data!.hypothesis_checks.map((c) => (
                      <li key={c}>{c}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </Row>
        )}

        {primary && r && (
          <Row label="Primary outcome">
            <p>
              <span className="font-mono text-xs">{primary.name}</span>{" "}
              <span className="text-ink-soft">
                · {primary.definition} · {primary.higher_is_better ? "higher" : "lower"} is better
              </span>
            </p>
            {comparisons.length > 0 ? (
              <ul className="mt-2 space-y-1.5">
                {comparisons.map((c) => (
                  <li key={c.condition} className="flex flex-wrap items-center gap-x-2 gap-y-1">
                    <span>
                      <span className="font-mono text-xs">{c.condition}</span> vs{" "}
                      <span className="font-mono text-xs">{c.baseline}</span>:{" "}
                      <span className="tabular-nums">
                        {num(c.difference)} (95% CI {num(c.ci_low)} to {num(c.ci_high)})
                        {c.p_value !== null && c.p_value !== undefined && `, p adjusted ${pValue(c.p_adjusted ?? c.p_value)}`}
                      </span>
                    </span>
                    <ComparisonVerdict c={c} />
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-1 text-ink-soft">No comparison with the baseline was made.</p>
            )}
            {(!r.results.complete || r.results.warnings.length > 0) && (
              <p className="mt-2 flex items-start gap-1.5 text-warn">
                <AlertTriangle className="mt-0.5 size-4 shrink-0" />
                <span>
                  {!r.results.complete && "Some planned runs have no results. "}
                  {r.results.warnings.length > 0 &&
                    `${r.results.warnings.length} warning${r.results.warnings.length === 1 ? "" : "s"} about these results. `}
                  {stageOf("coder") && (
                    <Link to={`${base}/${stageOf("coder")}?tab=results`} className={linkStyle}>
                      Check the results
                    </Link>
                  )}
                </span>
              </p>
            )}
          </Row>
        )}

        {review.data && (
          <Row label="Review">
            <p className="font-medium">
              {accepted ? "The reviewer recommended accepting the report" : "The reviewer left the report to your judgment"}
              {", and you approved it."}
            </p>
            {review.data.review.summary && <p className="mt-0.5 text-ink-soft">{review.data.review.summary}</p>}
          </Row>
        )}
      </dl>

      <div className="mt-5 flex flex-wrap gap-2">
        {stageOf("reviewer") && (
          <Link to={`${base}/${stageOf("reviewer")}?tab=final`} className={buttonStyles()}>
            Read the final report
          </Link>
        )}
        {(review.data || stageOf("writer")) && (
          <a
            href={`/api/runs/${encodeURIComponent(run.project)}/${encodeURIComponent(run.run_id)}/report/bundle`}
            className={buttonStyles({ variant: "secondary" })}
          >
            <Download /> Everything (.zip)
          </a>
        )}
        {primary && stageOf("coder") && (
          <Link to={`${base}/${stageOf("coder")}?tab=results`} className={buttonStyles({ variant: "ghost" })}>
            All results
          </Link>
        )}
      </div>
    </section>
  );
}
