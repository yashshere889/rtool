import { FilePlus2, Send, X } from "lucide-react";
import { useId, useMemo, useState } from "react";
import { useSearchParams } from "react-router";

import { ApiError } from "@/api/client";
import { useLiterature } from "@/api/hooks";
import type { OpenGate, PaperCardData, RunDetail, StageSummary } from "@/api/types";
import { CheckpointCard, useGateDecision } from "@/components/CheckpointCard";
import { FailureCard } from "@/components/FailureCard";
import { ReopenButton } from "@/components/ReopenDialog";
import { StatusBadge } from "@/components/StatusBadge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent } from "@/components/ui/dialog";
import { Input, Textarea } from "@/components/ui/field";
import { Empty, ErrorNote, Loading } from "@/components/ui/states";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { STAGES } from "@/lib/stages";

import { GapsTab } from "./GapsTab";
import { MapTab, type PaperFilter } from "./MapTab";
import { OverviewTab } from "./OverviewTab";
import { PaperCard, type Choice } from "./PaperCard";
import { SearchTab } from "./SearchTab";
import { UploadDialog } from "./UploadDialog";

type Show = "included" | "full_text" | "abstract" | "needs_upload" | "excluded" | "all";
type Sort = "relevance" | "year" | "citations";

const SHOW: Record<Show, [string, (p: PaperCardData) => boolean]> = {
  included: ["All included", (p) => p.included],
  full_text: ["Read in full", (p) => p.included && p.read === "full_text"],
  abstract: ["Abstract only", (p) => p.included && p.read === "abstract"],
  needs_upload: ["Need a PDF", (p) => p.needs_upload],
  excluded: ["Excluded at screening", (p) => !p.included],
  all: ["Everything found", () => true],
};

const SORT: Record<Sort, [string, (a: PaperCardData, b: PaperCardData) => number]> = {
  relevance: ["Most relevant", (a, b) => (b.relevance ?? 0) - (a.relevance ?? 0) || (b.citation_count ?? 0) - (a.citation_count ?? 0)],
  year: ["Newest", (a, b) => (b.year ?? 0) - (a.year ?? 0)],
  citations: ["Most cited", (a, b) => (b.citation_count ?? 0) - (a.citation_count ?? 0)],
};

/** The papers marked Keep or Drop on this page, sent with "request changes" (the agent applies them as marked). */
export function ChoicesBar({
  run,
  gate,
  choices,
  titles,
  onSent,
}: {
  run: RunDetail;
  gate: OpenGate;
  choices: Record<string, Choice>;
  titles: Map<string, string>;
  onSent: () => void;
}) {
  const { send, decide } = useGateDecision(run, gate);
  const [open, setOpen] = useState(false);
  const [comment, setComment] = useState("");
  const commentId = useId();
  const include = Object.keys(choices).filter((id) => choices[id] === "include");
  const exclude = Object.keys(choices).filter((id) => choices[id] === "exclude");
  return (
    <div className="sticky bottom-4 z-20 flex flex-wrap items-center gap-3 rounded-xl border border-accent/40 bg-surface px-4 py-3 shadow-lg">
      <span className="text-sm">
        You marked {include.length} to keep and {exclude.length} to drop.
      </span>
      <Button variant="ghost" size="sm" className="ml-auto" onClick={onSent}>
        Clear
      </Button>
      <Button size="sm" onClick={() => setOpen(true)}>
        <Send /> Send to the literature agent
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent
          title="Request changes to the literature"
          description="The papers are kept or dropped exactly as you marked them, and the agent runs again (downloading and reading what it now needs)."
        >
          <div className="space-y-3 text-sm">
            {include.length > 0 && (
              <div>
                <p className="font-medium">Keep</p>
                <ul className="list-disc pl-5 text-ink-soft">{include.map((id) => <li key={id}>{titles.get(id) ?? id}</li>)}</ul>
              </div>
            )}
            {exclude.length > 0 && (
              <div>
                <p className="font-medium">Drop</p>
                <ul className="list-disc pl-5 text-ink-soft">{exclude.map((id) => <li key={id}>{titles.get(id) ?? id}</li>)}</ul>
              </div>
            )}
            <label htmlFor={commentId} className="block font-medium">
              Anything else? (optional: with a comment the papers are also screened again)
            </label>
            <Textarea id={commentId} rows={3} value={comment} onChange={(e) => setComment(e.target.value)} placeholder="For example: also cover studies in older adults." />
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <DialogClose asChild>
              <Button variant="ghost">Cancel</Button>
            </DialogClose>
            <Button
              busy={decide.isPending}
              onClick={() =>
                send("reject", { feedback: comment.trim(), include, exclude }, "Request changes to the literature", {
                  onSuccess: () => {
                    setOpen(false);
                    onSent();
                  },
                })
              }
            >
              Send
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function Stat({ value, label }: { value: number; label: string }) {
  return (
    <div className="rounded-lg border border-line bg-surface px-4 py-3">
      <p className="text-xl font-semibold">{value}</p>
      <p className="text-xs text-ink-soft">{label}</p>
    </div>
  );
}

export function LiteraturePage({ run, stage }: { run: RunDetail; stage: StageSummary }) {
  const literature = useLiterature(run.project, run.run_id);
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") ?? "papers";
  const [show, setShow] = useState<Show>("included");
  const [sort, setSort] = useState<Sort>("relevance");
  const [text, setText] = useState("");
  const [filter, setFilter] = useState<PaperFilter | null>(null);
  const [adding, setAdding] = useState(false);
  const [choices, setChoices] = useState<Record<string, Choice>>({});
  const gate = stage.gate && !stage.gate.in_stage && stage.gate.kind === "approve" ? stage.gate : null;
  const queuedDecision = gate && run.queued_actions.some((a) => a.stage === gate.stage && a.gate === gate.gate);
  const choose = (id: string) => (choice: Choice | null) =>
    setChoices((prev) => {
      const next = { ...prev };
      if (choice === null) delete next[id];
      else next[id] = choice;
      return next;
    });

  const papers = useMemo(() => {
    const data = literature.data?.papers ?? [];
    const needle = text.trim().toLowerCase();
    return data
      .filter(SHOW[filter ? "all" : show][1])
      .filter((p) => !filter || filter.ids.includes(p.id))
      .filter((p) => !needle || `${p.title} ${p.authors.join(" ")} ${p.venue} ${p.bib_key}`.toLowerCase().includes(needle))
      .sort(SORT[sort][1]);
  }, [literature.data, show, sort, text, filter]);

  const setTab = (value: string) => setParams((prev) => {
    const next = new URLSearchParams(prev);
    next.set("tab", value);
    return next;
  });

  const header = (
    <header>
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold">Literature</h1>
        <StatusBadge stage={stage} run={run} />
        <div className="ml-auto">
          <ReopenButton
            run={run}
            stage={stage}
            label="Reopen the literature"
            title="Reopen the literature checkpoint?"
            description="To add papers you have, or to have the search and screening redone with your comments."
          />
        </div>
      </div>
      <p className="mt-2 max-w-3xl text-sm text-ink-soft">{STAGES.literature.description}</p>
    </header>
  );

  if (literature.isPending) return <div className="space-y-6">{header}<Loading label="Loading the papers" /></div>;
  if (literature.isError) {
    const missing = literature.error instanceof ApiError && literature.error.status === 404;
    return (
      <div className="space-y-6">
        {header}
        {stage.status === "failed" && <FailureCard run={run} stage={stage} />}
        {missing ? (
          <Empty title="No papers yet">The literature stage has not finished its first search. This page fills in when it has.</Empty>
        ) : (
          <ErrorNote error={literature.error} />
        )}
      </div>
    );
  }
  const view = literature.data;
  const stats = view.stats;

  return (
    <div className="space-y-6">
      {header}
      {stage.gate && <CheckpointCard run={run} gate={stage.gate} showStageLink={false} />}
      {stage.status === "failed" && <FailureCard run={run} stage={stage} />}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat value={stats.full_text ?? 0} label="papers read in full" />
        <Stat value={stats.abstract_only ?? 0} label="from the abstract only" />
        <Stat value={view.papers.filter((p) => p.needs_upload).length} label="need a PDF from you" />
        <Stat value={stats.verified_claims ?? 0} label="findings checked against a quote" />
      </div>

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="papers">Papers</TabsTrigger>
          <TabsTrigger value="map">Map</TabsTrigger>
          <TabsTrigger value="gaps">Gaps & open problems</TabsTrigger>
          <TabsTrigger value="overview">State of the art</TabsTrigger>
          <TabsTrigger value="search">Search the text</TabsTrigger>
        </TabsList>

        <TabsContent value="papers" className="mt-4 space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <Input
              aria-label="Filter by title, author or venue"
              placeholder="Filter by title, author or venue"
              className="w-full sm:w-72"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            <select
              aria-label="Which papers"
              className="h-9 rounded-lg border border-line bg-surface px-3 text-sm"
              value={show}
              onChange={(e) => {
                setShow(e.target.value as Show);
                setFilter(null);
              }}
            >
              {Object.entries(SHOW).map(([value, [label]]) => (
                <option key={value} value={value}>
                  {label} ({view.papers.filter(SHOW[value as Show][1]).length})
                </option>
              ))}
            </select>
            <select
              aria-label="Sort"
              className="h-9 rounded-lg border border-line bg-surface px-3 text-sm"
              value={sort}
              onChange={(e) => setSort(e.target.value as Sort)}
            >
              {Object.entries(SORT).map(([value, [label]]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
            <Button variant="ghost" size="sm" className="ml-auto" onClick={() => setAdding(true)}>
              <FilePlus2 /> Add a paper you have
            </Button>
          </div>
          {filter && (
            <p className="flex items-center gap-2 text-sm">
              Showing papers for <span className="font-medium">{filter.label}</span>
              <button type="button" className="inline-flex items-center gap-1 text-accent" onClick={() => setFilter(null)}>
                <X className="size-3.5" /> clear
              </button>
            </p>
          )}
          {papers.length === 0 ? (
            <Empty title="No papers match" />
          ) : (
            <div className="space-y-3">
              {papers.map((paper, i) => (
                <PaperCard
                  key={paper.id}
                  paper={paper}
                  run={run}
                  stage={view.stage}
                  anchor={`paper-${i}`}
                  choice={choices[paper.id] ?? null}
                  onChoose={gate && !queuedDecision ? choose(paper.id) : undefined}
                />
              ))}
            </div>
          )}
          {gate && !queuedDecision && Object.keys(choices).length > 0 && (
            <ChoicesBar
              run={run}
              gate={gate}
              choices={choices}
              titles={new Map(view.papers.map((p) => [p.id, p.title]))}
              onSent={() => setChoices({})}
            />
          )}
        </TabsContent>
        <TabsContent value="map" className="mt-4">
          <MapTab
            view={view}
            onFilter={(f) => {
              setFilter(f);
              setTab("papers");
            }}
          />
        </TabsContent>
        <TabsContent value="gaps" className="mt-4">
          <GapsTab view={view} run={run} />
        </TabsContent>
        <TabsContent value="overview" className="mt-4">
          <OverviewTab view={view} run={run} />
        </TabsContent>
        <TabsContent value="search" className="mt-4">
          <SearchTab run={run} stage={view.stage} />
        </TabsContent>
      </Tabs>
      {adding && <UploadDialog run={run} stage={view.stage} open={adding} onOpenChange={setAdding} />}
    </div>
  );
}
