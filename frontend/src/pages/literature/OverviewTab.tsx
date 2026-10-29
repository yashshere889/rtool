import { Fragment } from "react";
import { Link } from "react-router";

import { readerPath } from "@/api/client";
import type { LiteratureView, RunDetail } from "@/api/types";
import { Card, CardBody, CardHeader } from "@/components/ui/card";

/** Overview text in which [bib_key] citations become links to the papers. */
function Cited({ text, view, run }: { text: string; view: LiteratureView; run: RunDetail }) {
  const parts = text.split(/(\[[A-Za-z0-9_:.-]+\])/g);
  return (
    <>
      {parts.map((part, i) => {
        const key = part.startsWith("[") ? part.slice(1, -1) : "";
        const id = key ? view.keys[key] : undefined;
        if (!id) return <Fragment key={i}>{part}</Fragment>;
        const paper = view.papers.find((p) => p.id === id);
        return (
          <Link
            key={i}
            to={readerPath(run.project, run.run_id, view.stage, id)}
            title={paper ? `${paper.title} (${paper.year ?? "n.d."})` : key}
            className="rounded bg-accent-soft px-1 text-[0.85em] font-medium text-accent underline decoration-accent/40 underline-offset-2"
          >
            {key}
          </Link>
        );
      })}
    </>
  );
}

export function OverviewTab({ view, run }: { view: LiteratureView; run: RunDetail }) {
  const paragraphs = view.overview.split(/\n\s*\n/).filter((p) => p.trim());
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_18rem]">
      <Card>
        <CardHeader
          title="State of the art"
          description="Written by the AI model from the extracted facts; every citation links to the paper it comes from."
        />
        <CardBody className="space-y-4 text-[0.95rem] leading-relaxed">
          {paragraphs.length ? (
            paragraphs.map((p, i) => (
              <p key={i}>
                <Cited text={p} view={view} run={run} />
              </p>
            ))
          ) : (
            <p className="text-ink-soft">No overview was written.</p>
          )}
        </CardBody>
      </Card>
      <Card>
        <CardHeader title="How the papers were found" />
        <CardBody className="space-y-2 text-sm">
          <p className="text-ink-soft">Searches:</p>
          <ul className="list-inside list-disc space-y-1">
            {view.queries.map((q) => (
              <li key={q}>{q}</li>
            ))}
          </ul>
          <p className="pt-2 text-ink-soft">
            {view.stats.candidates ?? 0} candidates screened, {view.stats.included ?? 0} kept,{" "}
            {view.stats.verified_claims ?? 0} of {view.stats.claims ?? 0} extracted claims verified against the text.
          </p>
        </CardBody>
      </Card>
    </div>
  );
}
