import { useState } from "react";

import type { CoverageView, LiteratureView } from "@/api/types";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { cn } from "@/lib/cn";

export interface PaperFilter {
  label: string;
  ids: string[];
}

function Timeline({ timeline }: { timeline: Record<string, number> }) {
  const years = Object.keys(timeline).sort();
  if (!years.length) return <p className="text-sm text-ink-soft">No publication years recorded.</p>;
  const most = Math.max(...Object.values(timeline));
  return (
    <div className="flex h-28 items-end gap-1.5" role="img" aria-label="Papers per year">
      {years.map((year) => (
        <div key={year} className="flex flex-1 flex-col items-center gap-1" title={`${year}: ${timeline[year]} papers`}>
          <span className="text-[10px] text-ink-soft">{timeline[year]}</span>
          <div className="w-full max-w-8 rounded-t bg-accent/70" style={{ height: `${(timeline[year] / most) * 64}px` }} />
          <span className="text-[10px] text-ink-soft">{year.slice(2)}</span>
        </div>
      ))}
    </div>
  );
}

function Coverage({ cells, onFilter }: { cells: CoverageView[]; onFilter: (filter: PaperFilter) => void }) {
  const [picked, setPicked] = useState<CoverageView | null>(null);
  const methods = [...new Set(cells.map((c) => c.method))];
  const datasets = [...new Set(cells.map((c) => c.dataset))];
  if (!cells.length) return <p className="text-sm text-ink-soft">No method was linked to a dataset.</p>;
  const most = Math.max(...cells.map((c) => c.paper_ids.length));
  const cell = (m: string, d: string) => cells.find((c) => c.method === m && c.dataset === d);
  return (
    <div className="space-y-3">
      <div className="overflow-x-auto">
        <table className="text-xs">
          <thead>
            <tr>
              <th className="p-1 text-left font-medium text-ink-soft">Method \ data</th>
              {datasets.map((d) => (
                <th key={d} className="max-w-28 p-1 text-left align-bottom font-medium text-ink-soft">
                  {d}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {methods.map((m) => (
              <tr key={m}>
                <th className="max-w-48 p-1 text-left font-medium">{m}</th>
                {datasets.map((d) => {
                  const c = cell(m, d);
                  return (
                    <td key={d} className="p-0.5">
                      {c ? (
                        <button
                          type="button"
                          onClick={() => setPicked(c)}
                          className={cn("size-9 rounded font-semibold text-white", picked === c && "ring-2 ring-ink")}
                          style={{ backgroundColor: `color-mix(in srgb, var(--accent) ${30 + (70 * c.paper_ids.length) / most}%, transparent)` }}
                          aria-label={`${m} on ${d}: ${c.paper_ids.length} papers`}
                        >
                          {c.paper_ids.length}
                        </button>
                      ) : (
                        <span className="block size-9 rounded border border-dashed border-line" title="No paper tests this pairing" />
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-ink-soft">Empty squares are pairings no paper tested — possible gaps.</p>
      {picked && (
        <div className="rounded-lg bg-muted p-3 text-sm">
          <p className="font-medium">
            {picked.method} on {picked.dataset}
          </p>
          {picked.results.length > 0 && <p className="mt-1 text-ink-soft">Reported: {picked.results.join("; ")}</p>}
          <button
            type="button"
            className="mt-2 font-medium text-accent"
            onClick={() => onFilter({ label: `${picked.method} on ${picked.dataset}`, ids: picked.paper_ids })}
          >
            Show these {picked.paper_ids.length} papers
          </button>
        </div>
      )}
    </div>
  );
}

export function MapTab({ view, onFilter }: { view: LiteratureView; onFilter: (filter: PaperFilter) => void }) {
  const kinds = [
    ["method", "Methods"],
    ["dataset", "Data"],
    ["metric", "Measures"],
  ] as const;
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card className="lg:col-span-2">
        <CardHeader title="Topics" description="Groups of papers with similar content." />
        <CardBody className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {view.topics.map((topic) => (
            <button
              key={topic.label}
              type="button"
              onClick={() => onFilter({ label: `Topic: ${topic.label}`, ids: topic.paper_ids })}
              className="rounded-lg border border-line p-3 text-left hover:border-accent/50"
            >
              <p className="font-medium">{topic.label}</p>
              <p className="mt-1 text-sm text-ink-soft">{topic.summary}</p>
              <p className="mt-2 text-xs font-medium text-accent">{topic.paper_ids.length} papers →</p>
            </button>
          ))}
        </CardBody>
      </Card>
      <Card>
        <CardHeader title="Methods, data and measures" description="How many papers use each. Click to see them." />
        <CardBody className="space-y-4">
          {kinds.map(([kind, label]) => {
            const items = view.entities.filter((e) => e.kind === kind);
            return (
              <div key={kind}>
                <h4 className="mb-2 text-xs font-semibold tracking-wide text-ink-soft uppercase">{label}</h4>
                <div className="flex flex-wrap gap-1.5">
                  {items.length ? (
                    items.map((e) => (
                      <button
                        key={e.name}
                        type="button"
                        onClick={() => onFilter({ label: e.name, ids: e.paper_ids })}
                        className="rounded-full bg-muted px-2.5 py-0.5 text-xs hover:bg-accent-soft hover:text-accent"
                      >
                        {e.name} <span className="text-ink-soft">{e.paper_ids.length}</span>
                      </button>
                    ))
                  ) : (
                    <span className="text-sm text-ink-soft">None recorded.</span>
                  )}
                </div>
              </div>
            );
          })}
        </CardBody>
      </Card>
      <Card>
        <CardHeader title="Papers per year" />
        <CardBody>
          <Timeline timeline={view.timeline} />
        </CardBody>
      </Card>
      <Card className="lg:col-span-2">
        <CardHeader title="Which methods were tried on which data" description="Number of papers for each pairing." />
        <CardBody>
          <Coverage cells={view.coverage} onFilter={onFilter} />
        </CardBody>
      </Card>
    </div>
  );
}
