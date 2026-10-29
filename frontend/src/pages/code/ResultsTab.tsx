import { AlertTriangle, FileText } from "lucide-react";

import { ApiError, fileUrl } from "@/api/client";
import { useResults } from "@/api/hooks";
import type { ExperimentResults, RunDetail } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Empty, ErrorNote, Loading } from "@/components/ui/states";
import { num, pValue } from "@/lib/code";

import { Section } from "../plan/PlanRead";

type Comparison = ExperimentResults["comparisons"][number];

/** Why a comparison has no p-value (mirrors Comparison.verdict on the server). */
function untested(c: Comparison): string {
  if (c.n < 2) return "Too few runs to test.";
  if (c.constant_difference) return "The difference was identical for every seed: it may be a fixed offset, not a measured effect.";
  return "The results did not vary across seeds, so there is nothing to test against.";
}

/** The analysed results: each condition against the baseline, per metric, with the figures. */
export function ResultsTab({ run }: { run: RunDetail }) {
  const results = useResults(run.project, run.run_id);
  if (results.isPending) return <Loading label="Loading the results" />;
  if (results.isError) {
    if (results.error instanceof ApiError && results.error.status === 404) {
      return <Empty title="No results yet">They appear here when the runs have finished and been analysed.</Empty>;
    }
    return <ErrorNote error={results.error} />;
  }
  const { results: r, metrics, baseline } = results.data;
  const figureOf = (metric: string) => r.figures.find((f) => f.split("/").pop() === `${metric}.svg`);
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <Badge tone={r.complete ? "ok" : "warn"}>{r.complete ? "Every planned run finished" : "Some runs have no results"}</Badge>
        <span className="text-ink-soft">
          {r.runs.filter((x) => x.status === "ok").length} runs · {r.gpu_hours_requested} GPU-hours requested
        </span>
        <span className="ml-auto flex gap-3">
          {r.tables_path && (
            <a href={fileUrl(run.project, run.run_id, r.tables_path)} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent">
              <FileText className="size-4" /> Tables
            </a>
          )}
          <a href={fileUrl(run.project, run.run_id, r.results_path)} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent">
            <FileText className="size-4" /> results.json
          </a>
        </span>
      </div>
      {r.warnings.length > 0 && (
        <div role="note" aria-label="Warnings" className="rounded-lg border border-warn/40 bg-warn-soft/40 px-3 py-2 text-sm">
          <p className="flex items-center gap-1.5 font-medium text-warn">
            <AlertTriangle className="size-4 shrink-0" /> Check these results: something real may not have been measured
          </p>
          <ul className="mt-1 list-disc space-y-0.5 pl-5">
            {r.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </div>
      )}
      {r.notes.length > 0 && (
        <ul className="list-disc space-y-0.5 rounded-lg bg-info-soft py-2 pr-3 pl-8 text-sm text-info">
          {r.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}

      {metrics.map((metric) => {
        const comparisons = r.comparisons.filter((c) => c.metric === metric.name);
        const summaries = r.summaries.filter((s) => s.metric === metric.name);
        const figure = figureOf(metric.name);
        return (
          <Section key={metric.name} title={metric.name}>
            <p className="flex flex-wrap items-center gap-2 text-sm text-ink-soft">
              {metric.primary ? (
                <Badge tone="info">Primary outcome · confirmatory</Badge>
              ) : (
                <Badge>Exploratory</Badge>
              )}
              <span>
                {metric.definition} · {metric.higher_is_better ? "higher" : "lower"} is better
              </span>
            </p>
            {comparisons.length > 0 && (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[40rem] text-sm">
                  <caption className="pb-2 text-left text-xs font-medium text-ink-soft">Compared with the baseline ({baseline ?? "–"})</caption>
                  <thead>
                    <tr className="border-b border-line text-left text-xs text-ink-soft">
                      <th scope="col" className="py-2 pr-3 font-medium">Condition</th>
                      <th scope="col" className="py-2 pr-3 font-medium">Difference (95% CI)</th>
                      <th scope="col" className="py-2 pr-3 font-medium">p</th>
                      <th scope="col" className="py-2 pr-3 font-medium">p adjusted</th>
                      <th scope="col" className="py-2 pr-3 font-medium">Effect size d</th>
                      <th scope="col" className="py-2 font-medium">Verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparisons.map((c) => (
                      <tr key={c.condition} className="border-b border-line last:border-0">
                        <td className="py-2 pr-3 font-mono text-xs">{c.condition}</td>
                        <td className="py-2 pr-3 tabular-nums">
                          {num(c.difference)} ({num(c.ci_low)} to {num(c.ci_high)})
                        </td>
                        <td className="py-2 pr-3 tabular-nums" title={c.test}>
                          {pValue(c.p_value)}
                        </td>
                        <td className="py-2 pr-3 tabular-nums">{pValue(c.p_adjusted)}</td>
                        <td className="py-2 pr-3 tabular-nums">{num(c.effect_size)}</td>
                        <td className="py-2">
                          {c.p_value === null || c.p_value === undefined ? (
                            <span title={untested(c)}>
                              <Badge tone="warn">No test possible</Badge>
                            </span>
                          ) : !c.significant ? (
                            <Badge>No clear difference</Badge>
                          ) : c.better ? (
                            <Badge tone="ok">Better than baseline</Badge>
                          ) : (
                            <Badge tone="bad">Worse than baseline</Badge>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="mt-1 text-xs text-ink-soft">
                  {comparisons[0].test}, n = {comparisons[0].n} per condition
                  {comparisons[0].correction !== "none" && `; p adjusted with ${comparisons[0].correction === "holm" ? "Holm's" : "the Bonferroni"} method`}.
                </p>
                {comparisons
                  .filter((c) => c.p_value === null || c.p_value === undefined)
                  .map((c) => (
                    <p key={c.condition} className="mt-1 text-xs text-warn">
                      <span className="font-mono">{c.condition}</span>: {untested(c)}
                    </p>
                  ))}
              </div>
            )}
            <div className="grid items-start gap-4 lg:grid-cols-2">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-line text-left text-xs text-ink-soft">
                      <th scope="col" className="py-2 pr-3 font-medium">Condition</th>
                      <th scope="col" className="py-2 pr-3 font-medium">Mean</th>
                      <th scope="col" className="py-2 pr-3 font-medium">95% CI</th>
                      <th scope="col" className="py-2 font-medium">n</th>
                    </tr>
                  </thead>
                  <tbody>
                    {summaries.map((s) => (
                      <tr key={s.condition} className="border-b border-line last:border-0">
                        <td className="py-2 pr-3 font-mono text-xs">
                          {s.condition} {s.condition === baseline && <Badge tone="info">baseline</Badge>}
                        </td>
                        <td className="py-2 pr-3 tabular-nums">{num(s.mean)}</td>
                        <td className="py-2 pr-3 tabular-nums">
                          {num(s.ci_low)} to {num(s.ci_high)}
                        </td>
                        <td className="py-2 tabular-nums">{s.n}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {figure && (
                <img
                  src={fileUrl(run.project, run.run_id, figure)}
                  alt={`Mean ${metric.name} per condition with 95% confidence intervals`}
                  className="w-full rounded-lg border border-line bg-white p-2"
                />
              )}
            </div>
          </Section>
        );
      })}
    </div>
  );
}
