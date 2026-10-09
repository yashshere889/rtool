import { Link } from "react-router";

import { readerPath } from "@/api/client";
import type { LiteratureView, RunDetail } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Empty } from "@/components/ui/states";
import { GAP_KINDS } from "@/lib/literature";

function PaperLinks({ ids, view, run }: { ids: string[]; view: LiteratureView; run: RunDetail }) {
  const papers = ids.map((id) => view.papers.find((p) => p.id === id)).filter((p) => p !== undefined);
  if (!papers.length) return null;
  return (
    <ul className="mt-2 space-y-1 text-sm">
      {papers.map((p) => (
        <li key={p.id}>
          <Link to={readerPath(run.project, run.run_id, view.stage, p.id)} className="text-accent hover:underline">
            {p.title}
          </Link>{" "}
          <span className="text-ink-soft">({p.year ?? "n.d."})</span>
        </li>
      ))}
    </ul>
  );
}

export function GapsTab({ view, run }: { view: LiteratureView; run: RunDetail }) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader
          title="Open problems"
          description="What the papers' limitations and future-work statements point to (written by the AI model from verified quotes)."
        />
        <CardBody>
          {view.open_problems.length ? (
            <ol className="space-y-5">
              {view.open_problems.map((problem, i) => (
                <li key={problem.statement} className="flex gap-3">
                  <span className="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-full bg-accent-soft text-xs font-semibold text-accent">
                    {i + 1}
                  </span>
                  <div>
                    <p className="font-medium">{problem.statement}</p>
                    <p className="mt-1 text-sm text-ink-soft">Still open because {problem.why_open.charAt(0).toLowerCase() + problem.why_open.slice(1)}.</p>
                    <PaperLinks ids={problem.paper_ids} view={view} run={run} />
                  </div>
                </li>
              ))}
            </ol>
          ) : (
            <Empty title="No open problems were identified" />
          )}
        </CardBody>
      </Card>
      <Card>
        <CardHeader title="Gaps in the collection" description="Found by counting: pairings nobody tested, thin or dated topics." />
        <CardBody>
          {view.gaps.length ? (
            <ul className="space-y-4">
              {view.gaps.map((gap) => (
                <li key={gap.description}>
                  <div className="flex flex-wrap items-start gap-2">
                    <Badge tone="stale">{GAP_KINDS[gap.kind] ?? gap.kind}</Badge>
                    <span className="text-sm">{gap.description}</span>
                  </div>
                  <PaperLinks ids={gap.paper_ids} view={view} run={run} />
                </li>
              ))}
            </ul>
          ) : (
            <Empty title="No gaps were found by counting" />
          )}
        </CardBody>
      </Card>
    </div>
  );
}
