import { BookOpen, Upload } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router";

import { readerPath } from "@/api/client";
import { usePaper } from "@/api/hooks";
import type { PaperCardData, RunDetail } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { buttonStyles } from "@/components/ui/button";
import { ErrorNote, Loading } from "@/components/ui/states";
import { cn } from "@/lib/cn";
import { authorsShort, foundBy, paperStatus } from "@/lib/literature";

import { PaperPanel, type PanelKind } from "./PaperPanels";
import { UploadDialog } from "./UploadDialog";

const PANELS: { kind: PanelKind; label: (p: PaperCardData) => string }[] = [
  { kind: "summary", label: () => "Summary" },
  { kind: "methodology", label: () => "Methodology" },
  { kind: "findings", label: (p) => `Key findings${p.claims.result ? ` (${p.claims.result})` : ""}` },
  { kind: "limitations", label: (p) => `Limitations${p.claims.limitation ? ` (${p.claims.limitation})` : ""}` },
  { kind: "gaps", label: () => "Gaps & future work" },
  { kind: "datasets", label: () => "Datasets" },
  { kind: "references", label: () => "References & citations" },
  { kind: "cite", label: () => "Cite" },
];

function Relevance({ value, reason }: { value: number | null; reason: string }) {
  if (value == null) return null;
  return (
    <div className="flex items-center gap-2 text-xs text-ink-soft" title={reason}>
      <span className="flex gap-0.5" aria-hidden>
        {Array.from({ length: 10 }, (_, i) => (
          <span key={i} className={cn("h-2 w-1.5 rounded-sm", i < value ? "bg-accent" : "bg-line")} />
        ))}
      </span>
      <span>
        Relevance {value}/10{reason ? ` — ${reason}` : ""}
      </span>
    </div>
  );
}

export function PaperCard({ paper, run, stage, anchor }: { paper: PaperCardData; run: RunDetail; stage: string; anchor: string }) {
  const [panel, setPanel] = useState<PanelKind | null>(null);
  const [uploading, setUploading] = useState(false);
  const detail = usePaper(run.project, run.run_id, paper.id, panel !== null);
  const status = paperStatus(paper);
  const found = foundBy(paper);
  const ctx = { project: run.project, runId: run.run_id, stage };

  return (
    <article id={anchor} aria-labelledby={`${anchor}-title`} className="rounded-xl border border-line bg-surface p-4 shadow-sm sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <h3 id={`${anchor}-title`} className="text-base leading-snug font-semibold">
            {paper.title}
          </h3>
          <p className="mt-1 text-sm text-ink-soft">
            {authorsShort(paper.authors)} · {paper.year ?? "n.d."}
            {paper.venue ? ` · ${paper.venue}` : ""}
            {paper.citation_count != null ? ` · ${paper.citation_count.toLocaleString()} citations` : ""}
          </p>
        </div>
        <Badge tone={status.tone}>{status.label}</Badge>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1">
        <Relevance value={paper.relevance} reason={paper.relevance_reason} />
        {paper.topics.map((topic) => (
          <Badge key={topic}>{topic}</Badge>
        ))}
        {found && <span className="text-xs text-ink-soft">{found}</span>}
      </div>

      {paper.included && (
        <div className="mt-4 flex flex-wrap gap-2" role="toolbar" aria-label="About this paper">
          {paper.file ? (
            <Link to={readerPath(run.project, run.run_id, stage, paper.id)} className={buttonStyles({ size: "sm" })}>
              <BookOpen /> Open full paper
            </Link>
          ) : paper.needs_upload ? (
            <button type="button" className={buttonStyles({ size: "sm" })} onClick={() => setUploading(true)}>
              <Upload /> Upload PDF
            </button>
          ) : (
            <Link to={readerPath(run.project, run.run_id, stage, paper.id)} className={buttonStyles({ size: "sm", variant: "secondary" })}>
              <BookOpen /> Read abstract
            </Link>
          )}
          {PANELS.map(({ kind, label }) => (
            <button
              key={kind}
              type="button"
              aria-pressed={panel === kind}
              onClick={() => setPanel(panel === kind ? null : kind)}
              className={buttonStyles({ size: "sm", variant: panel === kind ? "primary" : "secondary" })}
            >
              {label(paper)}
            </button>
          ))}
        </div>
      )}

      {panel && (
        <div className="mt-4 rounded-lg border border-line bg-canvas/60 p-4">
          {detail.isPending && <Loading />}
          {detail.isError && <ErrorNote error={detail.error} />}
          {detail.data && <PaperPanel kind={panel} detail={detail.data} ctx={ctx} />}
        </div>
      )}
      {uploading && (
        <UploadDialog run={run} stage={stage} open={uploading} onOpenChange={setUploading} paperId={paper.id} initialTitle={paper.title} />
      )}
    </article>
  );
}
