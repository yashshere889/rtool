import { CircleAlert, CircleCheck, FileText, Pencil, Undo2 } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router";

import { ApiError, fileUrl } from "@/api/client";
import { readerPath } from "@/api/client";
import { useSynthesis, useUndo } from "@/api/hooks";
import type { RunDetail, StageSummary } from "@/api/types";
import { CheckpointCard } from "@/components/CheckpointCard";
import { FailureCard } from "@/components/FailureCard";
import { StatusBadge } from "@/components/StatusBadge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Empty, ErrorNote, Loading } from "@/components/ui/states";
import { count } from "@/lib/format";
import { STAGES } from "@/lib/stages";

import { Section } from "../plan/PlanRead";
import { ReviewEditor } from "./ReviewEditor";

/** The literature review: every sentence with the findings and quotes it rests on, and what was dropped. */
export function SynthesisPage({ run, stage }: { run: RunDetail; stage: StageSummary }) {
  const artifact = useSynthesis(run.project, run.run_id, stage.status !== "pending");
  const undo = useUndo(run.project, run.run_id);
  const [editing, setEditing] = useState(false);
  const literature = run.stages.find((s) => s.agent === "literature")?.id ?? "literature";
  const header = (
    <header>
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold">Literature review</h1>
        <StatusBadge stage={stage} run={run} />
      </div>
      <p className="mt-2 max-w-3xl text-sm text-ink-soft">{STAGES.synthesis.description}</p>
    </header>
  );
  const top = (
    <>
      {stage.gate && <CheckpointCard run={run} gate={stage.gate} showStageLink={false} />}
      {stage.status === "failed" && <FailureCard run={run} stage={stage} />}
    </>
  );
  if (artifact.isPending && stage.status !== "pending") return <div className="space-y-6">{header}{top}<Loading label="Loading the review" /></div>;
  if (artifact.isError || !artifact.data) {
    const missing = !artifact.error || (artifact.error instanceof ApiError && artifact.error.status === 404);
    return (
      <div className="space-y-6">
        {header}
        {top}
        {missing ? <Empty title="No review yet">It is written once the literature stage is approved.</Empty> : <ErrorNote error={artifact.error} />}
      </div>
    );
  }
  const view = artifact.data;
  const review = view.synthesis;
  const queuedEdit = run.queued_actions.find((a) => a.kind === "edit" && a.payload.artifact === "synthesis");
  const kept = review.sections.reduce((n, s) => n + s.sentences.length, 0);
  if (editing) {
    return (
      <div className="space-y-6">
        {header}
        <ReviewEditor run={run} view={view} onClose={() => setEditing(false)} />
      </div>
    );
  }
  return (
    <div className="space-y-6">
      {header}
      {top}
      {queuedEdit && (
        <div className="flex flex-wrap items-center gap-3 rounded-lg bg-info-soft px-4 py-3 text-sm text-info">
          Your edits to the review are on their way; they are applied when a worker picks them up.
          <Button variant="secondary" size="sm" busy={undo.isPending} onClick={() => undo.mutate(queuedEdit.id)}>
            <Undo2 /> Undo
          </Button>
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <Badge tone="ok">{count(kept, "sentence")} kept</Badge>
        <Badge tone={review.dropped.length ? "warn" : "neutral"}>{count(review.dropped.length, "sentence")} dropped</Badge>
        {view.by && <Badge tone="info">Edited by {view.by.replace(/ \(web\)$/, "")}</Badge>}
        {view.editable && !queuedEdit && (
          <Button variant="secondary" size="sm" className="ml-auto" onClick={() => setEditing(true)}>
            <Pencil /> Edit the review
          </Button>
        )}
        {review.markdown_path && (
          <a className="ml-auto inline-flex items-center gap-1 text-accent underline underline-offset-2" href={fileUrl(run.project, run.run_id, review.markdown_path)} target="_blank" rel="noreferrer">
            <FileText className="size-4" /> The review as a document
          </a>
        )}
      </div>

      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="space-y-4">
          {review.sections.map((section) => (
            <Section key={section.title} title={section.title}>
              <ul className="space-y-3">
                {section.sentences.map((s, i) => (
                  <li key={i} className="text-sm">
                    <span>{s.text}</span>{" "}
                    <span className="text-ink-soft">{s.papers.map((p) => `[@${p}]`).join(" ")}</span>
                    <details className="mt-1">
                      <summary className="cursor-pointer text-xs text-ink-soft">Rests on {s.claims.join(", ")}</summary>
                      <ul className="mt-2 space-y-2 border-l-2 border-line pl-3">
                        {s.claims.map((id) => {
                          const claim = review.claims[id];
                          return claim ? (
                            <li key={id} className="space-y-1">
                              <p className="text-xs">
                                <span className="font-mono">{id}</span> · {claim.text}
                              </p>
                              <blockquote className="text-xs text-ink-soft italic">“{claim.quote}”</blockquote>
                              <Link className="text-xs text-accent underline underline-offset-2" to={readerPath(run.project, run.run_id, literature, claim.paper_id, { quote: claim.quote })}>
                                Show in the paper
                              </Link>
                            </li>
                          ) : (
                            <li key={id} className="text-xs text-bad">
                              <span className="font-mono">{id}</span> was not among the findings offered
                            </li>
                          );
                        })}
                      </ul>
                    </details>
                  </li>
                ))}
              </ul>
            </Section>
          ))}
          {review.dropped.length > 0 && (
            <Section title="Sentences left out" id="dropped">
              <p className="text-sm text-ink-soft">The model wrote these, but a check failed, so they are not in the review.</p>
              <ul className="space-y-2 text-sm">
                {review.dropped.map((d, i) => (
                  <li key={i}>
                    <span className="line-through decoration-ink-soft/50">{d.text}</span>
                    <span className="block text-xs text-warn">{d.reason}</span>
                  </li>
                ))}
              </ul>
            </Section>
          )}
        </div>
        <aside className="space-y-2 rounded-xl border border-line bg-surface p-4 xl:sticky xl:top-20">
          <h2 className="text-sm font-semibold">Checks</h2>
          <ul className="space-y-1.5 text-sm">
            {review.checks.map((c) => (
              <li key={c.name} className={c.ok ? "flex gap-1.5 text-ok" : "flex gap-1.5 text-warn"}>
                {c.ok ? <CircleCheck className="mt-0.5 size-4 shrink-0" /> : <CircleAlert className="mt-0.5 size-4 shrink-0" />}
                <span>
                  <span className="text-ink">{c.name}</span>
                  {c.detail && <span className="block text-xs text-ink-soft">{c.detail}</span>}
                </span>
              </li>
            ))}
          </ul>
        </aside>
      </div>
    </div>
  );
}
