import { useState, type ReactNode } from "react";
import { Link } from "react-router";

import { readerPath } from "@/api/client";
import type { ClaimView, PaperDetail, PaperRef } from "@/api/types";
import { CopyButton, ExternalLinks } from "@/components/PaperLinks";
import { QuoteBlock } from "@/components/QuoteBlock";
import { Badge } from "@/components/ui/badge";
import { GAP_KINDS, plainCitation } from "@/lib/literature";

export type PanelKind =
  | "summary"
  | "methodology"
  | "findings"
  | "limitations"
  | "gaps"
  | "datasets"
  | "references"
  | "cite";

export interface PanelContext {
  project: string;
  runId: string;
  stage: string;
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-2">
      <h4 className="text-xs font-semibold tracking-wide text-ink-soft uppercase">{title}</h4>
      {children}
    </section>
  );
}

function Chips({ items, empty = "Not stated in the paper." }: { items: string[]; empty?: string }) {
  if (!items.length) return <p className="text-sm text-ink-soft">{empty}</p>;
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((item) => (
        <Badge key={item} tone="info">
          {item}
        </Badge>
      ))}
    </div>
  );
}

function Claims({ claims, ctx, paperId, empty }: { claims: ClaimView[]; ctx: PanelContext; paperId: string; empty: string }) {
  if (!claims.length) return <p className="text-sm text-ink-soft">{empty}</p>;
  return (
    <div className="space-y-4">
      {claims.map((claim) => (
        <QuoteBlock
          key={claim.index}
          claim={claim}
          showInPaper={readerPath(ctx.project, ctx.runId, ctx.stage, paperId, claim.index)}
        />
      ))}
    </div>
  );
}

function Provenance({ detail }: { detail: PaperDetail }) {
  return (
    <p className="text-xs text-ink-soft">
      {detail.paper.read === "abstract"
        ? "Only the abstract was available, so these facts come from it."
        : "Extracted from the full text by the literature agent. Findings are only kept with a quote that was found in the paper."}
    </p>
  );
}

function RefList({ refs, ctx, empty }: { refs: PaperRef[]; ctx: PanelContext; empty: string }) {
  if (!refs.length) return <p className="text-sm text-ink-soft">{empty}</p>;
  return (
    <ul className="space-y-1.5 text-sm">
      {refs.map((ref) => (
        <li key={ref.id}>
          <Link to={readerPath(ctx.project, ctx.runId, ctx.stage, ref.id)} className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent">
            {ref.title}
          </Link>{" "}
          <span className="text-ink-soft">
            ({ref.year ?? "n.d."}){ref.bib_key ? ` · ${ref.bib_key}` : ""}
          </span>
        </li>
      ))}
    </ul>
  );
}

function MethodsText({ text }: { text: string }) {
  const [all, setAll] = useState(false);
  if (!text) return <p className="text-sm text-ink-soft">The paper has no section the agent recognised as its methods.</p>;
  const long = text.length > 1200;
  return (
    <div>
      <p className="text-sm leading-relaxed whitespace-pre-line">{all || !long ? text : `${text.slice(0, 1200)}…`}</p>
      {long && (
        <button type="button" className="mt-1 text-sm font-medium text-accent" onClick={() => setAll(!all)}>
          {all ? "Show less" : "Show the whole section"}
        </button>
      )}
    </div>
  );
}

/** Text written by the model about the paper: labelled as such, with numbers it may have made up flagged. */
function AiWritten({ title, detail, children }: { title: string; detail: PaperDetail; children: ReactNode }) {
  const unverified = detail.digest?.unverified_numbers ?? [];
  return (
    <section className="space-y-2 rounded-lg border border-dashed border-line bg-muted/40 p-3">
      <h4 className="flex flex-wrap items-center gap-2 text-xs font-semibold tracking-wide text-ink-soft uppercase">
        {title}
        <Badge tone="info">AI-written · not checked against quotes</Badge>
      </h4>
      {children}
      {unverified.length > 0 && (
        <p className="text-xs text-warn">Contains numbers not found in the paper's text: {unverified.join(" ")}</p>
      )}
    </section>
  );
}

export function PaperPanel({ kind, detail, ctx }: { kind: PanelKind; detail: PaperDetail; ctx: PanelContext }) {
  const paper = detail.paper;
  const digest = detail.digest;
  const results = detail.claims.filter((c) => c.kind === "result");
  switch (kind) {
    case "summary":
      return (
        <div className="space-y-4">
          {digest && (
            <AiWritten title="In brief" detail={detail}>
              <p className="text-sm leading-relaxed">{digest.summary}</p>
            </AiWritten>
          )}
          <Section title="Research problem">
            <p className="text-sm leading-relaxed">{detail.problem || "Not stated."}</p>
          </Section>
          <Section title="What the authors did">
            <p className="text-sm leading-relaxed">{detail.approach || "Not stated."}</p>
          </Section>
          {results.length > 0 && (
            <Section title="Main findings">
              <Claims claims={results.slice(0, 3)} ctx={ctx} paperId={paper.id} empty="" />
            </Section>
          )}
          {paper.abstract && (
            <details className="text-sm">
              <summary className="cursor-pointer font-medium text-accent">Abstract</summary>
              <p className="mt-2 leading-relaxed text-ink-soft">{paper.abstract}</p>
            </details>
          )}
          <Provenance detail={detail} />
        </div>
      );
    case "methodology":
      return (
        <div className="space-y-4">
          {digest && (
            <AiWritten title="Study at a glance" detail={detail}>
              <dl className="grid gap-x-4 gap-y-1.5 text-sm sm:grid-cols-[max-content_1fr]">
                <dt className="font-medium text-ink-soft">Design</dt>
                <dd>{digest.design}</dd>
                <dt className="font-medium text-ink-soft">Sample</dt>
                <dd>{digest.sample}</dd>
                <dt className="font-medium text-ink-soft">Procedure</dt>
                <dd>{digest.procedure}</dd>
              </dl>
            </AiWritten>
          )}
          <Section title="Approach">
            <p className="text-sm leading-relaxed">{detail.approach || "Not stated."}</p>
          </Section>
          <div className="grid gap-4 sm:grid-cols-3">
            <Section title="Methods">
              <Chips items={detail.methods} />
            </Section>
            <Section title="Data">
              <Chips items={detail.datasets} />
            </Section>
            <Section title="Measures">
              <Chips items={detail.metrics} />
            </Section>
          </div>
          <Section title="The paper's methods section (verbatim)">
            <MethodsText text={detail.methods_section} />
          </Section>
        </div>
      );
    case "findings":
      return <Claims claims={results} ctx={ctx} paperId={paper.id} empty="No findings were extracted from this paper." />;
    case "limitations":
      return (
        <Claims
          claims={detail.claims.filter((c) => c.kind === "limitation")}
          ctx={ctx}
          paperId={paper.id}
          empty="The authors state no limitations the agent could find."
        />
      );
    case "gaps":
      return (
        <div className="space-y-5">
          {digest && digest.gaps.length > 0 && (
            <AiWritten title="Gaps this paper leaves" detail={detail}>
              <ul className="list-disc space-y-1 pl-5 text-sm">
                {digest.gaps.map((gap) => (
                  <li key={gap}>{gap}</li>
                ))}
              </ul>
            </AiWritten>
          )}
          <Section title="Future work the authors suggest">
            <Claims
              claims={detail.claims.filter((c) => c.kind === "future_work")}
              ctx={ctx}
              paperId={paper.id}
              empty="The authors suggest no future work."
            />
          </Section>
          <Section title="Open problems this paper points to">
            {detail.open_problems.length ? (
              <ul className="space-y-2 text-sm">
                {detail.open_problems.map((p) => (
                  <li key={p.statement}>
                    <p className="font-medium">{p.statement}</p>
                    <p className="text-ink-soft">Still open because {p.why_open.charAt(0).toLowerCase() + p.why_open.slice(1)}.</p>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-ink-soft">None of the open problems in the literature map cite this paper.</p>
            )}
          </Section>
          {detail.gaps.length > 0 && (
            <Section title="Gaps in the literature involving this paper">
              <ul className="space-y-2 text-sm">
                {detail.gaps.map((gap) => (
                  <li key={gap.description} className="flex flex-wrap items-start gap-2">
                    <Badge tone="stale">{GAP_KINDS[gap.kind] ?? gap.kind}</Badge>
                    <span>{gap.description}</span>
                  </li>
                ))}
              </ul>
            </Section>
          )}
        </div>
      );
    case "datasets":
      return (
        <div className="space-y-4">
          <Section title="Data used">
            <Chips items={detail.datasets} empty="The paper names no dataset or study population." />
          </Section>
          <Section title="Where it sits in the literature">
            {detail.coverage.length ? (
              <ul className="space-y-1.5 text-sm">
                {detail.coverage.map((cell) => (
                  <li key={`${cell.method}-${cell.dataset}`}>
                    <span className="font-medium">{cell.method}</span> on <span className="font-medium">{cell.dataset}</span>
                    <span className="text-ink-soft">
                      {" "}
                      · {cell.paper_ids.length} paper{cell.paper_ids.length === 1 ? "" : "s"}
                      {cell.results.length ? ` · ${cell.results.join("; ")}` : ""}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-ink-soft">No method and dataset pairing recorded for this paper.</p>
            )}
          </Section>
        </div>
      );
    case "references":
      return (
        <div className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <Section title={`Cites (in this collection: ${detail.cites.length})`}>
              <RefList refs={detail.cites} ctx={ctx} empty="Cites no other paper in this collection." />
            </Section>
            <Section title={`Cited by (in this collection: ${detail.cited_by.length})`}>
              <RefList refs={detail.cited_by} ctx={ctx} empty="No paper in this collection cites it." />
            </Section>
          </div>
          <p className="text-sm text-ink-soft">
            {paper.citation_count != null ? `${paper.citation_count.toLocaleString()} citations overall. ` : ""}
            The full reference list is in the paper itself.
          </p>
          <ExternalLinks links={paper.links} />
        </div>
      );
    case "cite":
      return (
        <div className="space-y-3">
          <p className="rounded-lg bg-muted px-3 py-2 text-sm">{plainCitation(paper)}</p>
          {detail.bibtex && (
            <pre className="max-h-56 overflow-auto rounded-lg bg-muted px-3 py-2 font-mono text-xs">{detail.bibtex}</pre>
          )}
          <div className="flex flex-wrap gap-2">
            <CopyButton text={plainCitation(paper)} label="Copy citation" />
            {detail.bibtex && <CopyButton text={detail.bibtex} label="Copy BibTeX" />}
          </div>
        </div>
      );
  }
}
