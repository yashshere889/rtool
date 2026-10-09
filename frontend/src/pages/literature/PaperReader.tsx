import { ChevronLeft, ExternalLink, FileText, Highlighter, Upload } from "lucide-react";
import { Fragment, useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";

import { fileUrl, runPath } from "@/api/client";
import { usePaper, usePaperText } from "@/api/hooks";
import type { PaperText } from "@/api/types";
import { useRunContext } from "@/components/layout/RunLayout";
import { ExternalLinks } from "@/components/PaperLinks";
import { Badge } from "@/components/ui/badge";
import { Button, buttonStyles } from "@/components/ui/button";
import { Card, CardBody } from "@/components/ui/card";
import { ErrorNote, Loading } from "@/components/ui/states";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/cn";
import { authorsShort, paperStatus } from "@/lib/literature";
import { stageTitle } from "@/lib/stages";

import { PaperPanel } from "./PaperPanels";
import { UploadDialog } from "./UploadDialog";

const FOCUS = -1; // the quote asked for in the address (claim numbers start at 0)

/** The text the agent read, with each verified claim's quote marked where it occurs (and the asked-for quote). */
function HighlightedText({ text, active }: { text: PaperText; active: number | null }) {
  useEffect(() => {
    if (active === null) return;
    const id = active === FOCUS ? "quote" : `claim-${active}`;
    document.getElementById(id)?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [active, text]);
  const focus = text.focus;
  return (
    <div className="space-y-6">
      {text.source === "abstract" && (
        <p className="rounded-lg bg-info-soft px-3 py-2 text-sm text-info">Only the abstract of this paper was available.</p>
      )}
      {text.sections.map((section, s) => {
        const claims = text.highlights.filter(
          (h) => h.section === s && !(focus && focus.section === s && h.start < focus.end && focus.start < h.end),
        );
        const marks = [...claims, ...(focus && focus.section === s ? [focus] : [])].sort((a, b) => a.start - b.start);
        const nodes = [];
        let cursor = 0;
        for (const mark of marks) {
          if (mark.start < cursor) continue;
          nodes.push(<Fragment key={`t${cursor}`}>{section.text.slice(cursor, mark.start)}</Fragment>);
          nodes.push(
            <mark
              key={`m${mark.claim}`}
              id={mark.claim === FOCUS ? "quote" : `claim-${mark.claim}`}
              className={cn(
                "rounded px-0.5 text-ink",
                active === mark.claim ? "bg-warn/40 ring-2 ring-warn" : "bg-warn-soft",
              )}
            >
              {section.text.slice(mark.start, mark.end)}
            </mark>,
          );
          cursor = mark.end;
        }
        nodes.push(<Fragment key="end">{section.text.slice(cursor)}</Fragment>);
        return (
          <section key={s}>
            <h3 className="mb-2 font-semibold">{section.title}</h3>
            <p className="text-[0.95rem] leading-relaxed whitespace-pre-line">{nodes}</p>
          </section>
        );
      })}
    </div>
  );
}

export function PaperReader() {
  const { stageId = "literature" } = useParams();
  const { run } = useRunContext();
  const [params] = useSearchParams();
  const paperId = params.get("paper") ?? "";
  const claimParam = params.get("claim");
  const quote = params.get("quote") ?? undefined;
  const claim = quote ? FOCUS : claimParam === null ? null : Number(claimParam);
  const detail = usePaper(run.project, run.run_id, paperId);
  const text = usePaperText(run.project, run.run_id, paperId, quote);
  const [mode, setMode] = useState<"original" | "text">(claim !== null ? "text" : "original");
  const [uploading, setUploading] = useState(false);
  useEffect(() => {
    if (claim !== null) setMode("text");
  }, [claim]);
  // opened from another page (e.g. a hypothesis's citations): go back there; uploads still go to the literature
  const from = run.stages.find((s) => s.id === stageId);
  const literatureStage = run.stages.find((s) => s.agent === "literature")?.id ?? stageId;

  const back = (
    <Link to={`${runPath(run.project, run.run_id)}/${stageId}`} className="inline-flex items-center gap-1 text-sm text-ink-soft hover:text-ink">
      <ChevronLeft className="size-4" /> {from && from.agent !== "literature" ? `Back to ${stageTitle(stageId)}` : "All papers"}
    </Link>
  );
  if (detail.isPending) return <div className="space-y-4">{back}<Loading label="Opening the paper" /></div>;
  if (detail.isError) return <div className="space-y-4">{back}<ErrorNote error={detail.error} title="Could not open the paper" /></div>;

  const paper = detail.data.paper;
  const status = paperStatus(paper);
  const file = paper.file ? fileUrl(run.project, run.run_id, paper.file) : null;
  const showing = file ? mode : "text";
  const ctx = { project: run.project, runId: run.run_id, stage: stageId };

  return (
    <div className="space-y-4">
      {back}
      <header className="space-y-2">
        <h1 className="text-xl leading-snug font-semibold sm:text-2xl">{paper.title}</h1>
        <p className="text-sm text-ink-soft">
          {authorsShort(paper.authors, 6)} · {paper.year ?? "n.d."}
          {paper.venue ? ` · ${paper.venue}` : ""}
          {paper.bib_key ? ` · ${paper.bib_key}` : ""}
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <Badge tone={status.tone}>{status.label}</Badge>
          <ExternalLinks links={paper.links} />
        </div>
      </header>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_26rem]">
        <div className="min-w-0 space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            {file && (
              <div className="inline-flex rounded-lg border border-line bg-surface p-0.5" role="group" aria-label="View">
                <button
                  type="button"
                  aria-pressed={showing === "original"}
                  onClick={() => setMode("original")}
                  className={cn("inline-flex items-center gap-1.5 rounded-md px-3 py-1 text-sm", showing === "original" ? "bg-muted font-medium" : "text-ink-soft")}
                >
                  <FileText className="size-4" /> {paper.file_kind === "pdf" ? "PDF" : "Web page"}
                </button>
                <button
                  type="button"
                  aria-pressed={showing === "text"}
                  onClick={() => setMode("text")}
                  className={cn("inline-flex items-center gap-1.5 rounded-md px-3 py-1 text-sm", showing === "text" ? "bg-muted font-medium" : "text-ink-soft")}
                >
                  <Highlighter className="size-4" /> Text with findings marked
                </button>
              </div>
            )}
            {file && (
              <a href={file} target="_blank" rel="noreferrer" className={buttonStyles({ variant: "ghost", size: "sm" })}>
                Open in a new tab <ExternalLink />
              </a>
            )}
            {paper.needs_upload && (
              <Button size="sm" onClick={() => setUploading(true)}>
                <Upload /> Upload the PDF
              </Button>
            )}
          </div>
          {showing === "original" && file ? (
            <iframe
              src={file}
              title={`Full text of ${paper.title}`}
              className="h-[78vh] w-full rounded-xl border border-line bg-white"
              sandbox={paper.file_kind === "html" ? "" : undefined}
            />
          ) : (
            <Card>
              <CardBody className="max-h-[78vh] overflow-y-auto py-5">
                {text.isPending && <Loading label="Loading the text" />}
                {text.isError && <ErrorNote error={text.error} />}
                {text.data && quote && !text.data.focus && (
                  <p className="mb-4 rounded-lg bg-warn-soft px-3 py-2 text-sm text-warn">
                    The quoted passage could not be found in the text the agent read: “{quote}”
                  </p>
                )}
                {text.data && <HighlightedText text={text.data} active={claim} />}
              </CardBody>
            </Card>
          )}
        </div>

        <aside className="min-w-0">
          <Card className="xl:sticky xl:top-20">
            <CardBody>
              <Tabs defaultValue={claim !== null && claim !== FOCUS ? "findings" : "summary"}>
                <TabsList>
                  <TabsTrigger value="summary">Summary</TabsTrigger>
                  <TabsTrigger value="findings">Findings</TabsTrigger>
                  <TabsTrigger value="methodology">Methods</TabsTrigger>
                  <TabsTrigger value="gaps">Gaps</TabsTrigger>
                  <TabsTrigger value="references">References</TabsTrigger>
                  <TabsTrigger value="cite">Cite</TabsTrigger>
                </TabsList>
                <div className="max-h-[70vh] overflow-y-auto pt-4">
                  {(["summary", "methodology", "gaps", "references", "cite"] as const).map((kind) => (
                    <TabsContent key={kind} value={kind}>
                      <PaperPanel kind={kind} detail={detail.data} ctx={ctx} />
                    </TabsContent>
                  ))}
                  <TabsContent value="findings" className="space-y-6">
                    <PaperPanel kind="findings" detail={detail.data} ctx={ctx} />
                    <div>
                      <h4 className="mb-3 text-xs font-semibold tracking-wide text-ink-soft uppercase">Limitations</h4>
                      <PaperPanel kind="limitations" detail={detail.data} ctx={ctx} />
                    </div>
                  </TabsContent>
                </div>
              </Tabs>
            </CardBody>
          </Card>
        </aside>
      </div>
      {uploading && (
        <UploadDialog
          run={run}
          stage={literatureStage}
          open={uploading}
          onOpenChange={setUploading}
          paperId={paper.id}
          initialTitle={paper.title}
        />
      )}
    </div>
  );
}
