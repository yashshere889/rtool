import { BadgeCheck, ChevronDown, CircleAlert, ScanSearch } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router";

import { readerPath } from "@/api/client";
import type { Hypothesis, HypothesisCardData } from "@/api/types";
import { CopyButton, ExternalLinks } from "@/components/PaperLinks";
import { Badge } from "@/components/ui/badge";
import { Tooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/cn";
import { COMPUTE, CRITERIA, ROLES } from "@/lib/hypotheses";
import { authorsShort, plainCitation } from "@/lib/literature";

export interface Ctx {
  project: string;
  runId: string;
  stage: string; // the hypotheses stage: the reader's "back" link returns here
  weights: Record<string, number>;
  budget: number | null;
}

/** A titled part of a hypothesis that can be folded away. */
export function Block({ title, children, open = true }: { title: ReactNode; children: ReactNode; open?: boolean }) {
  return (
    <details open={open} className="group border-t border-line pt-3">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-2 text-xs font-semibold tracking-wide text-ink-soft uppercase [&::-webkit-details-marker]:hidden">
        {title}
        <ChevronDown className="size-4 transition-transform group-open:rotate-180" aria-hidden />
      </summary>
      <div className="mt-2 space-y-3">{children}</div>
    </details>
  );
}

function Labelled({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <p className="text-xs font-medium text-ink-soft">{label}</p>
      <div className="mt-0.5 text-sm leading-relaxed">{children}</div>
    </div>
  );
}

function Chips({ items }: { items: string[] }) {
  if (!items.length) return <span className="text-ink-soft">None named</span>;
  return (
    <span className="flex flex-wrap gap-1.5">
      {items.map((item) => (
        <Badge key={item} tone="info" className="whitespace-normal">
          {item}
        </Badge>
      ))}
    </span>
  );
}

function scoreText(value: number | undefined): string {
  if (value === undefined) return "–";
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

/** The five rubric scores (median of the judges, 1 to 5) and how much each counts. */
export function Rubric({ h, weights }: { h: Hypothesis; weights: Record<string, number> }) {
  return (
    <dl className="space-y-1.5">
      {CRITERIA.map((c) => {
        const value = h.scores[c.key];
        return (
          <div key={c.key} className="grid grid-cols-[6.5rem_1fr_auto] items-center gap-2 text-sm">
            <dt>
              <Tooltip content={c.help}>
                <span className="cursor-help underline decoration-line decoration-dotted underline-offset-4">{c.label}</span>
              </Tooltip>
            </dt>
            <dd>
              <span role="img" aria-label={`${scoreText(value)} out of 5`} className="flex gap-0.5">
                {[1, 2, 3, 4, 5].map((i) => (
                  <span key={i} className={cn("h-2 flex-1 rounded-sm", value !== undefined && i <= value ? "bg-accent" : "bg-line")} />
                ))}
              </span>
            </dd>
            <dd className="text-xs text-ink-soft tabular-nums">
              {scoreText(value)}/5{weights[c.key] ? ` · counts ${Math.round(weights[c.key] * 100)}%` : ""}
            </dd>
          </div>
        );
      })}
    </dl>
  );
}

export function Novelty({ card, ctx }: { card: HypothesisCardData; ctx: Ctx }) {
  const novelty = card.hypothesis.novelty;
  if (!novelty) return <p className="text-sm text-ink-soft">The novelty check did not finish for this hypothesis.</p>;
  const closest = card.papers.find((p) => p.paper.id === novelty.closest_paper_id)?.paper;
  return (
    <div className="space-y-1 text-sm">
      <p className="flex flex-wrap items-center gap-2">
        {novelty.novel ? (
          <Badge tone="ok">New compared with the papers read · {novelty.score}/10</Badge>
        ) : (
          <Badge tone="warn">May repeat earlier work · {novelty.score}/10</Badge>
        )}
      </p>
      <p className="leading-relaxed">{novelty.overlap}</p>
      {closest && (
        <p className="text-ink-soft">
          Closest earlier work:{" "}
          <Link to={readerPath(ctx.project, ctx.runId, ctx.stage, closest.id)} className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent">
            {closest.title}
          </Link>{" "}
          ({closest.year ?? "n.d."})
        </p>
      )}
    </div>
  );
}

export function PapersUsed({ card, ctx }: { card: HypothesisCardData; ctx: Ctx }) {
  if (!card.papers.length) return <p className="text-sm text-ink-soft">No paper from the collection is linked to it.</p>;
  return (
    <ul className="space-y-2">
      {card.papers.map(({ paper, roles }) => (
        <li key={paper.id} className="rounded-lg border border-line bg-canvas/50 p-3">
          <Link to={readerPath(ctx.project, ctx.runId, ctx.stage, paper.id)} className="text-sm leading-snug font-medium hover:text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent">
            {paper.title}
          </Link>
          <p className="mt-0.5 text-xs text-ink-soft">
            {authorsShort(paper.authors, 2)} · {paper.year ?? "n.d."}
            {paper.venue ? ` · ${paper.venue}` : ""}
            {paper.citation_count != null ? ` · ${paper.citation_count.toLocaleString()} citations` : ""}
          </p>
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            {roles.map((role) => (
              <Tooltip key={role} content={ROLES[role].help}>
                <span>
                  <Badge tone={role === "evidence" ? "ok" : role === "closest" ? "warn" : "neutral"}>{ROLES[role].label}</Badge>
                </span>
              </Tooltip>
            ))}
            <ExternalLinks links={paper.links} className="flex flex-wrap gap-2 text-xs" />
          </div>
        </li>
      ))}
    </ul>
  );
}

export function Citations({ card, ctx }: { card: HypothesisCardData; ctx: Ctx }) {
  const evidence = card.hypothesis.evidence;
  if (!evidence.length) return <p className="text-sm text-ink-soft">It cites no passage.</p>;
  return (
    <div className="space-y-3">
      {evidence.map((e, i) => {
        const paper = card.papers.find((p) => p.paper.id === e.paper_id)?.paper;
        const match = `${Math.round(e.support * 100)}% match`;
        return (
          <figure key={i} className="space-y-1.5">
            <blockquote className="border-l-2 border-accent/40 pl-3 text-sm text-ink-soft italic">“{e.quote}”</blockquote>
            <figcaption className="flex flex-wrap items-center gap-2 text-xs">
              {e.verified ? (
                <Tooltip content="The passage was found in the paper's text, so this citation is real.">
                  <span>
                    <Badge tone="ok">
                      <BadgeCheck className="size-3.5" /> Found in the paper ({match})
                    </Badge>
                  </span>
                </Tooltip>
              ) : (
                <Tooltip content="The passage could not be found in the paper. Check it before relying on it.">
                  <span>
                    <Badge tone="bad">
                      <CircleAlert className="size-3.5" /> Not found in the paper ({match})
                    </Badge>
                  </span>
                </Tooltip>
              )}
              <span className="text-ink-soft">
                {paper ? `${authorsShort(paper.authors, 1)} (${paper.year ?? "n.d."})` : e.paper_id}
              </span>
              {paper && (
                <Link
                  to={readerPath(ctx.project, ctx.runId, ctx.stage, paper.id, { quote: e.quote })}
                  className="inline-flex items-center gap-1 font-medium text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent"
                >
                  <ScanSearch className="size-3.5" /> Show in paper
                </Link>
              )}
            </figcaption>
          </figure>
        );
      })}
    </div>
  );
}

export function References({ card }: { card: HypothesisCardData }) {
  const lines = card.papers.map((p) => plainCitation(p.paper));
  if (!lines.length) return <p className="text-sm text-ink-soft">No references.</p>;
  return (
    <div className="space-y-3">
      <ol className="list-decimal space-y-1.5 pl-5 text-sm leading-snug">
        {lines.map((line, i) => (
          <li key={card.papers[i].paper.id}>{line}</li>
        ))}
      </ol>
      <div className="flex flex-wrap gap-2">
        <CopyButton text={lines.join("\n")} label="Copy references" />
        {card.bibtex && <CopyButton text={card.bibtex} label="Copy BibTeX" />}
      </div>
    </div>
  );
}

/** Everything about one hypothesis below its statement: design, novelty, scores, papers, citations, references. */
export function HypothesisDetails({ card, ctx, compact = false }: { card: HypothesisCardData; ctx: Ctx; compact?: boolean }) {
  const h = card.hypothesis;
  const overBudget = ctx.budget !== null && card.gpu_hours > ctx.budget;
  const notes = [...h.feasibility_notes, ...h.risks];
  return (
    <div className="space-y-3">
      <Block title="Design" open={!compact}>
        <Labelled label="Null hypothesis">{h.null_hypothesis}</Labelled>
        <Labelled label="Why it follows from the literature">{h.rationale}</Labelled>
        <Labelled label="Varied (independent)">
          <Chips items={h.independent_variables} />
        </Labelled>
        <Labelled label="Measured (dependent)">
          <Chips items={h.dependent_variables} />
        </Labelled>
        {h.controls.length > 0 && (
          <Labelled label="Kept the same (controls)">
            <Chips items={h.controls} />
          </Labelled>
        )}
        <Labelled label="Data">
          <Chips items={h.datasets} />
        </Labelled>
        <Labelled label="Methods">
          <Chips items={h.methods} />
        </Labelled>
        <Labelled label="Success criteria">{h.success_criteria}</Labelled>
      </Block>
      <Block title="Novelty" open={!compact}>
        <Novelty card={card} ctx={ctx} />
      </Block>
      <Block title="Feasibility & risks" open={!compact}>
        <p className="text-sm">
          <span className="font-medium">{COMPUTE[h.compute].label}</span>: {COMPUTE[h.compute].detail}
          {overBudget && <span className="text-warn"> — more than this run's GPU budget ({ctx.budget} GPU-hours)</span>}
        </p>
        {notes.length > 0 ? (
          <ul className="list-disc space-y-0.5 pl-5 text-sm">
            {notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-ink-soft">No risks noted.</p>
        )}
      </Block>
      <Block title="How it scored" open={!compact}>
        <Rubric h={h} weights={ctx.weights} />
        {h.rubric_comments.map((comment) => (
          <p key={comment} className="text-sm text-ink-soft">
            Judge: “{comment}”
          </p>
        ))}
      </Block>
      <Block title={`Papers used (${card.papers.length})`} open={!compact}>
        <PapersUsed card={card} ctx={ctx} />
      </Block>
      <Block title={`Citations (${h.evidence.length})`} open={!compact}>
        <Citations card={card} ctx={ctx} />
      </Block>
      <Block title={`References (${card.papers.length})`} open={!compact}>
        <References card={card} />
      </Block>
    </div>
  );
}
