import { Check, FileText, Pencil } from "lucide-react";
import { useState } from "react";
import { useSearchParams } from "react-router";

import { ApiError, fileUrl } from "@/api/client";
import { useHypotheses } from "@/api/hooks";
import type { HypothesesView, HypothesisCardData, Hypothesis, RunDetail, StageSummary } from "@/api/types";
import { CheckpointCard } from "@/components/CheckpointCard";
import { FailureCard } from "@/components/FailureCard";
import { ReopenButton } from "@/components/ReopenDialog";
import { StatusBadge } from "@/components/StatusBadge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Empty, ErrorNote, Loading } from "@/components/ui/states";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/cn";
import { dateTime, timeAgo } from "@/lib/format";
import { COMPUTE, fieldLabel, score100 } from "@/lib/hypotheses";
import { STAGES } from "@/lib/stages";

import { ChooseDialog } from "./ChooseDialog";
import { CompareTable } from "./CompareTable";
import { HypothesisDetails, type Ctx } from "./HypothesisDetails";

function show(value: unknown): string {
  return Array.isArray(value) ? value.join("; ") : String(value ?? "");
}

/** The hypothesis as it was chosen: the researcher's edits on top of what the agent offered. */
function asChosen(card: HypothesisCardData, view: HypothesesView): HypothesisCardData {
  const selection = view.selection;
  if (!selection?.edited) return card;
  const edits = Object.fromEntries(selection.changes.map((c) => [c.field, c.after])) as Partial<Hypothesis>;
  return { ...card, hypothesis: { ...card.hypothesis, ...edits } };
}

function Column({
  card,
  view,
  ctx,
  chosen,
  pending,
  onChoose,
}: {
  card: HypothesisCardData;
  view: HypothesesView;
  ctx: Ctx;
  chosen: boolean;
  pending: boolean;
  onChoose?: (editing: boolean) => void;
}) {
  const shown = chosen ? asChosen(card, view) : card;
  const h = shown.hypothesis;
  const edits = chosen ? (view.selection?.changes ?? []) : [];
  const title = `hypothesis-${card.index}`;
  return (
    <article
      aria-labelledby={title}
      className={cn(
        "flex min-w-0 flex-col rounded-xl border bg-surface shadow-sm",
        chosen ? "border-ok ring-2 ring-ok/20" : "border-line",
      )}
    >
      <div className="flex-1 space-y-3 p-4 sm:p-5">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="info">Option {(card.index ?? 0) + 1}</Badge>
          {chosen && (
            <Badge tone="ok">
              <Check className="size-3.5" /> Your choice
            </Badge>
          )}
          {pending && <Badge tone="warn">Your choice is on its way</Badge>}
          {edits.length > 0 && <Badge tone="info">Edited by you</Badge>}
          <span className="ml-auto text-sm text-ink-soft" title="Weighted rubric score; unverified citations lower it">
            Score <span className="font-semibold text-ink tabular-nums">{score100(h)}</span>/100
          </span>
        </div>
        <h2 id={title} className="text-lg leading-snug font-semibold">
          {h.title}
        </h2>
        <p className="rounded-lg bg-accent-soft/60 px-3 py-2.5 text-[0.95rem] leading-relaxed">{h.statement}</p>
        <p className="text-xs text-ink-soft">
          {COMPUTE[h.compute].label} · {h.evidence.length} citations · {card.papers.length} papers
        </p>
        {edits.length > 0 && (
          <div className="rounded-lg border border-info/30 bg-info-soft/50 p-3 text-sm">
            <p className="font-medium">Your edits</p>
            <ul className="mt-1 space-y-1">
              {edits.map((change) => (
                <li key={change.field}>
                  <span className="font-medium">{fieldLabel(change.field)}:</span> {show(change.after)}
                  <span className="block text-xs text-ink-soft">was: {show(change.before)}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        <HypothesisDetails card={shown} ctx={ctx} />
      </div>
      {onChoose && (
        <div className="sticky bottom-0 flex flex-wrap gap-2 rounded-b-xl border-t border-line bg-surface p-4">
          <Button onClick={() => onChoose(false)}>Choose this one</Button>
          <Button variant="secondary" onClick={() => onChoose(true)}>
            <Pencil /> Edit & choose
          </Button>
        </div>
      )}
    </article>
  );
}

function ChoiceBanner({ run, stage, view }: { run: RunDetail; stage: StageSummary; view: HypothesesView }) {
  const selection = view.selection;
  if (!selection) return null;
  const older = selection.from_version !== null && selection.from_version !== view.version;
  return (
    <section aria-label="Your choice" className="rounded-xl border border-ok/40 bg-ok-soft/40 p-4 sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs font-semibold tracking-wide text-ok uppercase">Your choice</p>
          <p className="mt-1 font-semibold">
            Option {selection.index + 1}: {selection.title}
          </p>
          <p className="mt-1 text-sm text-ink-soft" title={dateTime(selection.created_at)}>
            {selection.by ? `Chosen by ${selection.by.replace(/ \(web\)$/, "")}` : "Chosen automatically"} ·{" "}
            {timeAgo(selection.created_at)}
            {selection.edited ? ` · with your edits to ${selection.changes.map((c) => fieldLabel(c.field)).join(", ")}` : ""}
            {older ? ` · from an earlier set of hypotheses (set ${selection.from_version})` : ""}
          </p>
          {selection.note && <p className="mt-1 text-sm">Note for the planner: “{selection.note}”</p>}
        </div>
        <ReopenButton
          run={run}
          stage={stage}
          label="Change my choice"
          title="Change your choice?"
          description="The hypotheses are offered again. Choose another, edit one, or keep your current choice."
        />
      </div>
    </section>
  );
}

export function HypothesesPage({ run, stage }: { run: RunDetail; stage: StageSummary }) {
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") ?? "side";
  const versionParam = params.get("version");
  const hypotheses = useHypotheses(run.project, run.run_id, versionParam ? Number(versionParam) : undefined);
  const [choosing, setChoosing] = useState<{ card: HypothesisCardData; editing: boolean } | null>(null);

  const setParam = (key: string, value: string | null) =>
    setParams((prev) => {
      const next = new URLSearchParams(prev);
      if (value === null) next.delete(key);
      else next.set(key, value);
      return next;
    });

  const header = (
    <header>
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold">Hypotheses</h1>
        <StatusBadge stage={stage} run={run} />
      </div>
      <p className="mt-2 max-w-3xl text-sm text-ink-soft">{STAGES.hypotheses.description}</p>
    </header>
  );

  if (hypotheses.isPending) return <div className="space-y-6">{header}<Loading label="Loading the hypotheses" /></div>;
  if (hypotheses.isError) {
    const missing = hypotheses.error instanceof ApiError && hypotheses.error.status === 404;
    return (
      <div className="space-y-6">
        {header}
        {stage.status === "failed" && <FailureCard run={run} stage={stage} />}
        {missing ? (
          <Empty title="No hypotheses yet">
            They appear here once the literature is approved and this stage has drafted and checked them.
          </Empty>
        ) : (
          <ErrorNote error={hypotheses.error} />
        )}
      </div>
    );
  }

  const view = hypotheses.data;
  const latest = view.version === Math.max(...view.versions);
  const gate = stage.gate?.kind === "select" ? stage.gate : null;
  const queued = gate ? run.queued_actions.find((a) => a.stage === gate.stage && a.gate === gate.gate) : undefined;
  const queuedIndex = queued?.kind === "select" ? (queued.payload.index as number | undefined) : undefined;
  const chosenIndex = view.selection && view.selection.from_version === view.version ? view.selection.index : null;
  const canChoose = Boolean(gate && latest && !queued);
  const ctx: Ctx = {
    project: run.project,
    runId: run.run_id,
    stage: stage.id,
    weights: view.weights,
    budget: view.budget_gpu_hours,
  };

  return (
    <div className="space-y-6">
      {header}
      {stage.gate && <CheckpointCard run={run} gate={stage.gate} showStageLink={false} choiceElsewhere={Boolean(gate)} />}
      {stage.status === "failed" && <FailureCard run={run} stage={stage} />}
      {stage.status === "stale" && stage.stale_reason && (
        <p className="rounded-lg bg-stale-soft px-4 py-3 text-sm text-stale">Will run again: {stage.stale_reason}</p>
      )}
      {!gate && <ChoiceBanner run={run} stage={stage} view={view} />}

      <div className="flex flex-wrap items-center gap-3 text-sm">
        <span className="text-ink-soft">
          {view.shortlist.length} offered from {view.shortlist.length + view.others.length} drafted
        </span>
        {view.markdown && (
          <a
            href={fileUrl(run.project, run.run_id, view.markdown)}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent"
          >
            <FileText className="size-4" /> The agent's write-up
          </a>
        )}
        {view.versions.length > 1 && (
          <label className="ml-auto flex items-center gap-2">
            <span className="text-ink-soft">Set of hypotheses</span>
            <select
              className="h-8 rounded-md border border-line bg-surface px-2"
              value={view.version}
              onChange={(e) => {
                const value = Number(e.target.value);
                setParam("version", value === Math.max(...view.versions) ? null : String(value));
              }}
            >
              {[...view.versions].reverse().map((v, i) => (
                <option key={v} value={v}>
                  Set {v}
                  {i === 0 ? " (latest)" : ""}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {!latest && (
        <p className="rounded-lg bg-info-soft px-4 py-3 text-sm text-info">
          You are looking at an earlier set. Choices are made from the latest set.
        </p>
      )}

      <Tabs value={tab} onValueChange={(value) => setParam("tab", value === "side" ? null : value)}>
        <TabsList>
          <TabsTrigger value="side">Side by side</TabsTrigger>
          <TabsTrigger value="compare">Compare</TabsTrigger>
          <TabsTrigger value="others">Other drafts ({view.others.length})</TabsTrigger>
        </TabsList>
        <TabsContent value="side" className="mt-4">
          <div className={cn("grid items-start gap-4", view.shortlist.length >= 3 ? "xl:grid-cols-3" : "lg:grid-cols-2")}>
            {view.shortlist.map((card) => (
              <Column
                key={card.hypothesis.id}
                card={card}
                view={view}
                ctx={ctx}
                chosen={card.index === chosenIndex}
                pending={queuedIndex === card.index}
                onChoose={canChoose ? (editing) => setChoosing({ card, editing }) : undefined}
              />
            ))}
          </div>
        </TabsContent>
        <TabsContent value="compare" className="mt-4">
          <CompareTable cards={view.shortlist} chosen={chosenIndex} />
        </TabsContent>
        <TabsContent value="others" className="mt-4 space-y-3">
          <p className="max-w-3xl text-sm text-ink-soft">
            Everything else the agent drafted, with why it was not offered. These cannot be chosen directly; ask for new
            hypotheses with your comments if one of them is closer to what you want.
          </p>
          {view.others.length === 0 ? (
            <Empty title="Nothing else was drafted" />
          ) : (
            view.others.map((card) => (
              <article key={card.hypothesis.id} className="rounded-xl border border-line bg-surface p-4 sm:p-5">
                <div className="flex flex-wrap items-center gap-2">
                  {card.hypothesis.rejected_reason ? (
                    <Badge tone="bad">Set aside: {card.hypothesis.rejected_reason}</Badge>
                  ) : (
                    <Badge>Ranked lower · score {score100(card.hypothesis)}/100</Badge>
                  )}
                </div>
                <h3 className="mt-2 font-semibold">{card.hypothesis.title}</h3>
                <p className="mt-1 text-sm leading-relaxed">{card.hypothesis.statement}</p>
                <div className="mt-3">
                  <HypothesisDetails card={card} ctx={ctx} compact />
                </div>
              </article>
            ))
          )}
        </TabsContent>
      </Tabs>

      {choosing && gate && (
        <ChooseDialog
          run={run}
          gate={gate}
          card={choosing.card}
          editing={choosing.editing}
          open
          onOpenChange={(open) => !open && setChoosing(null)}
        />
      )}
    </div>
  );
}
