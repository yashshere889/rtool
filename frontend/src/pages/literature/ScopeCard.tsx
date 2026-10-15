import { CircleAlert, CircleCheck } from "lucide-react";

import type { LiteratureView } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardBody, CardHeader } from "@/components/ui/card";

import type { PaperFilter } from "./MapTab";

/** How the agent read the topic before searching, and what was measured about the result: what to check first. */
export function ScopeCard({ view, onFilter }: { view: LiteratureView; onFilter: (filter: PaperFilter) => void }) {
  const { scope, quality, subtopics } = view;
  if (!scope && !quality.length) return null;
  const warnings = quality.filter((q) => !q.ok);
  const passed = quality.filter((q) => q.ok);
  return (
    <Card>
      <CardHeader
        title={scope ? "How the topic was read" : "How the literature was gathered"}
        description={
          scope
            ? "Written by the AI model before searching; every paper was screened against the key concepts. If this is not what you meant, request changes with a comment saying what you meant."
            : undefined
        }
      />
      <CardBody className="space-y-4 text-sm">
        {scope && (
          <>
            <p className="text-[0.95rem] leading-relaxed">{scope.interpretation}</p>
            <dl className="grid gap-x-4 gap-y-3 sm:grid-cols-[9rem_minmax(0,1fr)]">
              <dt className="text-ink-soft">Key concepts</dt>
              <dd className="flex flex-wrap gap-1.5">
                {scope.concepts.map((c) => (
                  <Badge key={c} tone="info">
                    {c}
                  </Badge>
                ))}
              </dd>
              {scope.out_of_scope.length > 0 && (
                <>
                  <dt className="text-ink-soft">Left out</dt>
                  <dd>{scope.out_of_scope.join("; ")}</dd>
                </>
              )}
              {scope.other_readings.length > 0 && (
                <>
                  <dt className="text-ink-soft">Other readings</dt>
                  <dd>
                    <ul className="list-disc space-y-0.5 pl-5">
                      {scope.other_readings.map((r) => (
                        <li key={r}>{r}</li>
                      ))}
                    </ul>
                  </dd>
                </>
              )}
              {subtopics.length > 0 && (
                <>
                  <dt className="text-ink-soft">Sub-topics</dt>
                  <dd className="flex flex-wrap gap-1.5">
                    {subtopics.map((s) => (
                      <button
                        key={s.name}
                        type="button"
                        onClick={() => onFilter({ label: `Sub-topic: ${s.name}`, ids: s.paper_ids })}
                        className="rounded-full bg-muted px-2.5 py-0.5 text-xs hover:bg-accent-soft hover:text-accent"
                      >
                        {s.name} <span className={s.paper_ids.length <= 1 ? "font-semibold text-warn" : "text-ink-soft"}>{s.paper_ids.length}</span>
                      </button>
                    ))}
                  </dd>
                </>
              )}
            </dl>
          </>
        )}
        {quality.length > 0 && (
          <section aria-label="Quality checks">
            <h3 className="font-medium">
              Quality checks{" "}
              <span className="font-normal text-ink-soft">{warnings.length ? `(${warnings.length} to look at)` : "(all passed)"}</span>
            </h3>
            {warnings.length > 0 && (
              <ul className="mt-2 space-y-1.5">
                {warnings.map((q) => (
                  <li key={q.check} className="flex gap-2">
                    <CircleAlert aria-hidden className="mt-0.5 size-4 shrink-0 text-warn" />
                    <span>
                      <span className="sr-only">Look at this: </span>
                      <span className="font-medium">{q.check}</span>: {q.value}
                      {q.note && <span className="block text-ink-soft">{q.note}.</span>}
                    </span>
                  </li>
                ))}
              </ul>
            )}
            {passed.length > 0 && (
              <details className="mt-2">
                <summary className="cursor-pointer text-ink-soft hover:text-ink">
                  {passed.length} {passed.length === 1 ? "check" : "checks"} passed
                </summary>
                <ul className="mt-2 space-y-1.5">
                  {passed.map((q) => (
                    <li key={q.check} className="flex gap-2">
                      <CircleCheck aria-hidden className="mt-0.5 size-4 shrink-0 text-ok" />
                      <span>
                        <span className="sr-only">Passed: </span>
                        <span className="font-medium">{q.check}</span>: {q.value}
                      </span>
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </section>
        )}
      </CardBody>
    </Card>
  );
}
