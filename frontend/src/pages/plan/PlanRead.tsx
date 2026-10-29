import { ExternalLink } from "lucide-react";
import type { ReactNode } from "react";

import type { ClusterView, ExperimentPlan } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/cn";
import { primaryMetric, RESOURCE_KINDS, RESOURCE_SOURCES, resourceCheck, resourceLink } from "@/lib/plan";

export function Section({ title, children, id }: { title: string; children: ReactNode; id?: string }) {
  return (
    <section id={id} aria-labelledby={id ? `${id}-title` : undefined} className="space-y-3 rounded-xl border border-line bg-surface p-4 sm:p-5">
      <h2 id={id ? `${id}-title` : undefined} className="text-base font-semibold">
        {title}
      </h2>
      {children}
    </section>
  );
}

function Table({ head, rows }: { head: string[]; rows: ReactNode[][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[32rem] text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs text-ink-soft">
            {head.map((h) => (
              <th key={h} scope="col" className="py-2 pr-3 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((cells, i) => (
            <tr key={i} className="border-b border-line align-top last:border-0">
              {cells.map((cell, j) => (
                <td key={j} className="py-2 pr-3">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const label = (pairs: [string, string][], value: string) => pairs.find(([v]) => v === value)?.[1] ?? value;

/** The GPU-hours a plan needs against the run's budget, as a bar. */
export function Budget({ hours, budget }: { hours: number; budget: number | null }) {
  if (budget === null) return <p className="text-sm">About {hours.toLocaleString()} GPU-hours (no budget set).</p>;
  const share = budget > 0 ? Math.min(1, hours / budget) : 1;
  const over = hours > budget;
  return (
    <div className="space-y-1">
      <div className="h-2 overflow-hidden rounded-full bg-muted" aria-hidden>
        <div className={cn("h-full rounded-full", over ? "bg-bad" : share > 0.8 ? "bg-warn" : "bg-ok")} style={{ width: `${share * 100}%` }} />
      </div>
      <p className={cn("text-sm", over && "font-medium text-bad")}>
        About {hours.toLocaleString()} of the run's {budget.toLocaleString()} GPU-hours
        {over ? ": over budget" : ""}
      </p>
    </div>
  );
}

export function ComputeSummary({ plan, cluster }: { plan: ExperimentPlan; cluster: ClusterView }) {
  const c = plan.compute;
  const partition = cluster.partitions.find((p) => p.name === c.partition);
  return (
    <div className="space-y-3">
      <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
        <div>
          <dt className="text-xs text-ink-soft">Where</dt>
          <dd>
            {c.partition}
            {partition?.note ? <span className="text-ink-soft"> ({partition.note})</span> : null}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-ink-soft">Each run</dt>
          <dd>
            {c.gpus_per_run ? `${c.gpus_per_run} × ${c.gpu_type ?? "GPU"} (${c.gpu_memory_gb} GB needed)` : "CPU only"} ·{" "}
            about {c.hours_per_run} h, time limit {c.time_limit}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-ink-soft">Runs</dt>
          <dd>
            {c.runs} ({plan.conditions.length + plan.ablations.length} conditions × {plan.seeds.length} seeds)
          </dd>
        </div>
        {c.justification && (
          <div>
            <dt className="text-xs text-ink-soft">Why</dt>
            <dd>{c.justification}</dd>
          </div>
        )}
      </dl>
      <Budget hours={c.gpu_hours_total} budget={cluster.budget_gpu_hours} />
    </div>
  );
}

export function ResourceTable({ plan }: { plan: ExperimentPlan }) {
  return (
    <Table
      head={["Resource", "Where from", "Purpose", "Checked", "Licence / size"]}
      rows={plan.resources.map((r) => {
        const link = resourceLink(r);
        const check = resourceCheck(r);
        return [
          <span key="n">
            <span className="font-medium">{r.name}</span>
            <span className="block text-xs text-ink-soft">{label(RESOURCE_KINDS, r.kind)}</span>
          </span>,
          <span key="s" className="break-all">
            <span className="block text-xs text-ink-soft">{label(RESOURCE_SOURCES, r.source)}</span>
            {link ? (
              <a href={link} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent">
                {r.identifier} <ExternalLink className="size-3" />
              </a>
            ) : (
              r.identifier
            )}
            {(r.subset || r.split) && (
              <span className="block text-xs text-ink-soft">{[r.subset && `subset ${r.subset}`, r.split && `split ${r.split}`].filter(Boolean).join(", ")}</span>
            )}
          </span>,
          r.purpose,
          <span key="c">
            <Badge tone={check.tone}>{check.label}</Badge>
            {r.note && <span className="mt-1 block text-xs text-ink-soft">{r.note}</span>}
          </span>,
          [r.license, r.size_gb != null ? `${r.size_gb} GB` : ""].filter(Boolean).join(" · ") || "–",
        ];
      })}
    />
  );
}

/** The plan as a document. */
export function PlanRead({ plan, cluster, title }: { plan: ExperimentPlan; cluster: ClusterView; title: string }) {
  return (
    <div className="space-y-4">
      <Section title="What it tests">
        {title && <p className="font-medium">{title}</p>}
        <p className="leading-relaxed">{plan.hypothesis}</p>
        <p className="text-sm text-ink-soft">Null hypothesis: {plan.null_hypothesis}</p>
        <p className="text-sm">
          <span className="font-medium">Objective.</span> {plan.objective}
        </p>
        <p className="text-sm leading-relaxed">
          <span className="font-medium">Design.</span> {plan.design}
        </p>
      </Section>

      <Section title="Variables">
        <Table
          head={["", "Name", "Description", "Levels"]}
          rows={[
            ...plan.independent_variables.map((v) => ["Varied", v.name, v.description, v.values.join(", ") || "–"]),
            ...plan.dependent_variables.map((v) => ["Measured", v.name, v.description, "–"]),
          ]}
        />
        {plan.controls.length > 0 && (
          <p className="text-sm">
            <span className="font-medium">Kept the same:</span> {plan.controls.join("; ")}
          </p>
        )}
      </Section>

      <Section title="Conditions">
        <Table
          head={["Condition", "What happens"]}
          rows={[
            ...plan.conditions.map((c) => [
              <span key="n" className="font-mono text-xs">
                {c.name} {c.is_baseline && <Badge tone="info">baseline</Badge>}
              </span>,
              c.description,
            ]),
            ...plan.ablations.map((c) => [
              <span key="n" className="font-mono text-xs">
                {c.name} <Badge>ablation</Badge>
              </span>,
              c.description,
            ]),
          ]}
        />
      </Section>

      <Section title="Measurements and statistics">
        <Table
          head={["Metric", "Definition", "Better"]}
          rows={plan.metrics.map((m) => [
            <span key="n" className="font-mono text-xs">
              {m.name} {m === primaryMetric(plan) && <Badge tone="info">primary outcome</Badge>}
            </span>,
            m.definition,
            m.higher_is_better ? "higher" : "lower",
          ])}
        />
        <Table
          head={["Test", "Comparison", "Alpha", "Correction"]}
          rows={plan.statistical_tests.map((t) => [t.test, t.comparison, String(t.alpha), t.correction])}
        />
        <p className="text-sm">
          {plan.seeds.length} independent runs (seeds) per condition.
        </p>
      </Section>

      <Section title="Procedure">
        <ol className="list-decimal space-y-1 pl-5 text-sm">
          {plan.procedure.map((step, i) => (
            <li key={i}>{step}</li>
          ))}
        </ol>
      </Section>

      <Section title="Datasets, models and software">
        <ResourceTable plan={plan} />
      </Section>

      <Section title="Compute">
        <ComputeSummary plan={plan} cluster={cluster} />
      </Section>

      <Section title="Expected results and risks">
        <p className="text-sm leading-relaxed">{plan.expected_results}</p>
        {plan.risks.length > 0 && (
          <ul className="list-disc space-y-0.5 pl-5 text-sm">
            {plan.risks.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        )}
      </Section>
    </div>
  );
}
