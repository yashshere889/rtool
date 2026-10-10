import { useQueryClient } from "@tanstack/react-query";
import { CircleAlert, CircleCheck, Download, FileText, Lock, Pencil, Printer, Quote, Sigma, Undo2 } from "lucide-react";
import { lazy, Suspense, useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, useSearchParams } from "react-router";

import { ApiError, fileUrl, readerPath } from "@/api/client";
import { keys, useDecide, useDiscardDraft, useDraft, useImpact, useReport, useReportCheck, useSaveDraft, useUndo } from "@/api/hooks";
import type { PaperInfo, ReportView, ReportWhich, RunDetail, SectionView } from "@/api/types";
import type { MarkdownEditorHandle } from "@/components/code/CodeEditor";
import { Markdown } from "@/components/Markdown";
import { Badge } from "@/components/ui/badge";
import { Button, buttonStyles } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Input, Textarea } from "@/components/ui/field";
import { Spinner } from "@/components/ui/spinner";
import { Empty, ErrorNote, Loading } from "@/components/ui/states";
import { Tooltip } from "@/components/ui/tooltip";
import { sameJson } from "@/lib/equal";
import { dateTime, timeAgo } from "@/lib/format";
import { ORIGINS, sectionHeadings, type ReportDraft } from "@/lib/report";
import { stageTitle } from "@/lib/stages";
import { useDebounced } from "@/lib/useDebounced";

const MarkdownEditor = lazy(() => import("@/components/code/CodeEditor").then((m) => ({ default: m.MarkdownEditor })));

function Rail({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-2 rounded-xl border border-line bg-surface p-4">
      <h2 className="text-sm font-semibold">{title}</h2>
      {children}
    </section>
  );
}

function OriginBadge({ origin }: { origin: string }) {
  const info = ORIGINS[origin] ?? ORIGINS.llm;
  return (
    <Tooltip content={info.help}>
      <span className="print:hidden">
        <Badge tone={info.tone}>{info.label}</Badge>
      </span>
    </Tooltip>
  );
}

/** The report as a document: sections, figures, linked citations and the reference list. */
function ReportDocument({ run, view, stage, papers }: { run: RunDetail; view: ReportView; stage: string; papers: Map<string, PaperInfo> }) {
  const headings = sectionHeadings(view.sections);
  return (
    <article className="space-y-6 rounded-xl border border-line bg-surface p-5 sm:p-8 print:border-0 print:p-0">
      <h1 className="text-2xl leading-tight font-semibold sm:text-3xl">{view.title}</h1>
      {view.sections.map((section) => (
        <section key={section.id} id={`section-${section.id}`} aria-labelledby={`heading-${section.id}`} className="scroll-mt-24">
          <div className="flex flex-wrap items-center gap-2">
            <h2 id={`heading-${section.id}`} className="text-lg font-semibold">
              {headings[section.id]}
            </h2>
            <OriginBadge origin={section.origin} />
          </div>
          <Markdown text={section.rendered} papers={papers} project={run.project} runId={run.run_id} stage={stage} baseDir={view.base_dir} />
        </section>
      ))}
      {view.references.length > 0 && (
        <section aria-labelledby="references">
          <h2 id="references" className="text-lg font-semibold">
            References
          </h2>
          <ol className="mt-2 list-decimal space-y-1.5 pl-5 text-sm">
            {view.references.map((paper) => (
              <li key={paper.key}>
                {paper.authors || "Anon."} ({paper.year ?? "n.d."}).{" "}
                <Link to={readerPath(run.project, run.run_id, stage, paper.paper_id)} className="text-accent hover:underline">
                  {paper.title}
                </Link>
                <span className="text-ink-soft"> [{paper.key}]</span>
              </li>
            ))}
          </ol>
        </section>
      )}
    </article>
  );
}

function Picker({
  open,
  onClose,
  title,
  items,
  onPick,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  items: { key: string; label: string; detail: string; insert: string }[];
  onPick: (text: string) => void;
}) {
  const [query, setQuery] = useState("");
  const id = useId();
  const needle = query.trim().toLowerCase();
  const shown = items.filter((i) => !needle || `${i.key} ${i.label} ${i.detail}`.toLowerCase().includes(needle)).slice(0, 80);
  return (
    <Dialog open={open} onOpenChange={(next) => !next && onClose()}>
      <DialogContent title={title} description="Inserted where the cursor was in the section you edited last.">
        <label htmlFor={id} className="sr-only">
          Search
        </label>
        <Input id={id} autoFocus placeholder="Search" value={query} onChange={(e) => setQuery(e.target.value)} />
        <ul className="mt-3 max-h-[55vh] space-y-1 overflow-y-auto">
          {shown.map((item) => (
            <li key={item.key}>
              <button
                type="button"
                onClick={() => {
                  onPick(item.insert);
                  onClose();
                }}
                className="w-full rounded-lg px-3 py-2 text-left text-sm hover:bg-muted"
              >
                <span className="block font-medium">{item.label}</span>
                <span className="block text-xs text-ink-soft">{item.detail}</span>
              </button>
            </li>
          ))}
          {shown.length === 0 && <li className="px-3 py-2 text-sm text-ink-soft">Nothing matches.</li>}
        </ul>
      </DialogContent>
    </Dialog>
  );
}

function toValue(view: ReportView, draft: ReportDraft) {
  return {
    title: draft.title,
    sections: view.sections.map((s) => ({ id: s.id, title: s.title, origin: s.origin, template: draft.templates[s.id] ?? s.template })),
  };
}

function SectionEditor({
  section,
  heading,
  value,
  problems,
  editor,
  onChange,
  onFocus,
  placeholders,
  papers,
}: {
  section: SectionView;
  heading: string;
  value: string;
  problems: string[];
  editor: (handle: MarkdownEditorHandle | null) => void;
  onChange: (text: string) => void;
  onFocus: () => void;
  placeholders: Map<string, string>;
  papers: Map<string, string>;
}) {
  if (!section.editable) {
    return (
      <section className="rounded-xl border border-line bg-canvas/40 p-4">
        <p className="flex items-center gap-2 text-sm font-semibold">
          <Lock className="size-4 text-ink-soft" /> {heading}
        </p>
        <p className="mt-1 text-sm text-ink-soft">{ORIGINS.rtool.help}</p>
      </section>
    );
  }
  return (
    <section className="space-y-2 rounded-xl border border-line bg-surface p-4">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-semibold">{heading}</h3>
        <OriginBadge origin={value === section.template ? section.origin : "human"} />
      </div>
      <Suspense fallback={<Loading label="Opening the editor" />}>
        <MarkdownEditor ref={editor} value={value} onChange={onChange} onFocus={onFocus} label={heading} placeholders={placeholders} papers={papers} />
      </Suspense>
      {problems.length > 0 && (
        <ul className="space-y-1 text-sm text-bad">
          {problems.map((p) => (
            <li key={p} className="flex gap-1.5">
              <CircleAlert className="mt-0.5 size-4 shrink-0" /> {p}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/** A report (or the final report) to read, print, download and edit. */
export function ReportWorkspace({ run, which, stage }: { run: RunDetail; which: ReportWhich; stage: string }) {
  const [params, setParams] = useSearchParams();
  const editing = params.get("mode") === "edit";
  const versionParam = params.get("version");
  const report = useReport(run.project, run.run_id, which, versionParam ? Number(versionParam) : undefined);
  const saved = useDraft(run.project, run.run_id, which);
  const { mutate: saveDraft, isPending: savingDraft } = useSaveDraft(run.project, run.run_id, which);
  const discardDraft = useDiscardDraft(run.project, run.run_id, which);
  const decide = useDecide(run.project, run.run_id);
  const undo = useUndo(run.project, run.run_id);
  const queryClient = useQueryClient();
  const [draft, setDraft] = useState<ReportDraft | null>(null);
  const [note, setNote] = useState("");
  const [picker, setPicker] = useState<"result" | "citation" | null>(null);
  const editors = useRef<Record<string, MarkdownEditorHandle | null>>({});
  const [active, setActive] = useState<string | null>(null);
  const ids = { title: useId(), note: useId() };

  const view = report.data;
  const queued = run.queued_actions.find((a) => a.kind === "edit" && a.payload.artifact === which);
  const canEdit = Boolean(view?.editable) && !queued;
  const impact = useImpact(run.project, run.run_id, view?.stage ?? stage, editing && canEdit);

  const setMode = (mode: string | null) =>
    setParams((prev) => {
      const next = new URLSearchParams(prev);
      if (mode) next.set("mode", mode);
      else next.delete("mode");
      return next;
    });

  // start from the saved draft (if made on the latest version) or from the report
  useEffect(() => {
    if (!editing || !canEdit || draft || !view || saved.isPending) return;
    const stored = saved.data;
    if (stored && stored.base_version === view.head) {
      setDraft(stored.value as unknown as ReportDraft);
      setNote(stored.note);
    } else {
      setDraft({ title: view.title, templates: Object.fromEntries(view.sections.map((s) => [s.id, s.template])) });
    }
  }, [editing, canEdit, draft, view, saved.isPending, saved.data]);

  const changed = useMemo(
    () => (draft && view ? view.sections.filter((s) => (draft.templates[s.id] ?? s.template) !== s.template) : []),
    [draft, view],
  );
  const titleChanged = Boolean(draft && view && draft.title !== view.title);
  const dirty = changed.length > 0 || titleChanged;
  const value = useMemo(() => (draft && view ? toValue(view, draft) : null), [draft, view]);
  const settled = useDebounced(value, 700);
  const check = useReportCheck(run.project, run.run_id, which, view?.head ?? 0, settled, Boolean(editing && dirty && settled));
  const pending = useMemo(
    () => (editing && dirty && view && draft ? { base_version: view.head, value: draft, note } : null),
    [editing, dirty, view, draft, note],
  );
  const toSave = useDebounced(pending, 1200);
  useEffect(() => {
    if (toSave) saveDraft(toSave as never);
  }, [toSave, saveDraft]);

  const placeholderMap = useMemo(() => new Map((view?.placeholders ?? []).map((p) => [p.key, `${p.text} — ${p.meaning}`])), [view]);
  const paperTitles = useMemo(() => new Map((view?.papers ?? []).map((p) => [p.key, `${p.title} (${p.year ?? "n.d."})`])), [view]);
  const papers = useMemo(() => new Map((view?.papers ?? []).map((p) => [p.key, p])), [view]);

  if (report.isPending) return <Loading label="Loading the report" />;
  if (report.isError || !view) {
    if (report.error instanceof ApiError && report.error.status === 404) {
      return <Empty title={which === "report" ? "No report yet" : "No final report yet"}>It appears here once the {which === "report" ? "writer" : "reviewer"} has finished.</Empty>;
    }
    return <ErrorNote error={report.error} />;
  }

  const headings = sectionHeadings(view.sections);
  const checked = check.data?.ok ? new Map(check.data.sections.map((s) => [s.id, s.problems])) : null;
  const typedNumbers = check.data?.ok ? check.data.findings.filter((f) => f.kind === "number" && changed.some((s) => s.id === f.section)) : [];
  const redo = (impact.data ?? []).filter((a) => a.effect !== "not_started");

  function apply() {
    if (!value || !view) return;
    decide.mutate(
      { request: { kind: "edit", stage: view.stage, payload: { artifact: which, base_version: view.head, value, note: note.trim() } }, label: "Your report edits" },
      {
        onSuccess: () => {
          setDraft(null);
          setNote("");
          queryClient.setQueryData(keys.draft(run.project, run.run_id, which), null);
          setMode(null);
        },
      },
    );
  }

  function discard() {
    discardDraft.mutate();
    setDraft(null);
    setNote("");
    setMode(null);
  }

  const toolbar = (
    <div className="flex flex-wrap items-center gap-2 print:hidden">
      <p className="mr-auto text-sm text-ink-soft" title={dateTime(view.created_at)}>
        Version {view.version}
        {view.by ? ` · edited by ${view.by.replace(/ \(web\)$/, "")}` : ` · by the ${which === "report" ? "writer" : "reviewer"}`} ·{" "}
        {timeAgo(view.created_at)}
        {view.edit_note && <span className="block">“{view.edit_note}”</span>}
      </p>
      {view.versions.length > 1 && (
        <select
          aria-label="Version"
          className="h-8 rounded-md border border-line bg-surface px-2 text-sm"
          value={view.version}
          onChange={(e) =>
            setParams((prev) => {
              const next = new URLSearchParams(prev);
              if (Number(e.target.value) === view.head) next.delete("version");
              else next.set("version", e.target.value);
              next.delete("mode");
              return next;
            })
          }
        >
          {[...view.versions].reverse().map((v) => (
            <option key={v} value={v}>
              Version {v}
              {v === view.head ? " (latest)" : ""}
            </option>
          ))}
        </select>
      )}
      <a href={fileUrl(run.project, run.run_id, view.markdown_path)} target="_blank" rel="noreferrer" className={buttonStyles({ variant: "ghost", size: "sm" })}>
        <FileText /> Markdown
      </a>
      <a href={fileUrl(run.project, run.run_id, view.bibtex_path)} target="_blank" rel="noreferrer" className={buttonStyles({ variant: "ghost", size: "sm" })}>
        <Quote /> BibTeX
      </a>
      <a href={`/api/runs/${encodeURIComponent(run.project)}/${encodeURIComponent(run.run_id)}/report/bundle`} className={buttonStyles({ variant: "ghost", size: "sm" })}>
        <Download /> Everything (.zip)
      </a>
      <Button variant="ghost" size="sm" onClick={() => window.print()}>
        <Printer /> Print or save as PDF
      </Button>
      {canEdit && !editing && (
        <Button size="sm" onClick={() => setMode("edit")}>
          <Pencil /> Edit
        </Button>
      )}
    </div>
  );

  const notices = (
    <>
      {queued && (
        <div className="flex flex-wrap items-center gap-3 rounded-lg bg-info-soft px-4 py-3 text-sm text-info print:hidden">
          Your edits are on their way; they are applied when a worker picks them up.
          <Button variant="secondary" size="sm" busy={undo.isPending} onClick={() => undo.mutate(queued.id)}>
            <Undo2 /> Undo and keep editing
          </Button>
        </div>
      )}
      {saved.data && saved.data.base_version !== view.head && !queued && view.version === view.head && (
        <div className="flex flex-wrap items-center gap-3 rounded-lg bg-warn-soft px-4 py-3 text-sm text-warn print:hidden">
          You have unsent edits made on version {saved.data.base_version}; the report has changed since.
          <Button variant="secondary" size="sm" onClick={() => discardDraft.mutate()}>
            Discard them
          </Button>
        </div>
      )}
      {!view.editable && view.why_not_editable && <p className="text-sm text-ink-soft print:hidden">Editing: {view.why_not_editable}.</p>}
    </>
  );

  if (!editing || !draft || !canEdit) {
    return (
      <div className="space-y-4">
        {toolbar}
        {notices}
        <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_18rem] print:block">
          <ReportDocument run={run} view={view} stage={stage} papers={papers} />
          <aside className="space-y-4 xl:sticky xl:top-20 print:hidden">
            <Rail title={`Notes from the ${which === "report" ? "writer's audit" : "writer and reviewer"} (${view.findings.length})`}>
              {view.findings.length === 0 ? (
                <p className="flex items-center gap-1.5 text-sm text-ok">
                  <CircleCheck className="size-4" /> Nothing to note.
                </p>
              ) : (
                <ul className="space-y-2 text-sm">
                  {view.findings.map((f, i) => (
                    <li key={i}>
                      <Badge tone={f.severity === "major" ? "bad" : "warn"}>{f.severity}</Badge>{" "}
                      {f.section && (
                        <a href={`#section-${f.section}`} className="text-accent hover:underline">
                          {headings[f.section] ?? f.section}
                        </a>
                      )}{" "}
                      {f.detail}
                    </li>
                  ))}
                </ul>
              )}
            </Rail>
            <Rail title="How the numbers get there">
              <p className="text-sm text-ink-soft">
                Results enter the text only as placeholders filled in from the analysis ({view.placeholders.length} available),
                and citations only from papers in the collection, so every number and reference can be traced.
              </p>
            </Rail>
          </aside>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {toolbar}
      {notices}
      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="min-w-0 space-y-3">
          <div className="flex flex-wrap items-end gap-3 rounded-xl border border-line bg-surface p-4">
            <div className="min-w-0 flex-1">
              <label htmlFor={ids.title} className="mb-1 block text-sm font-medium">
                Title
              </label>
              <Input id={ids.title} value={draft.title} onChange={(e) => setDraft({ ...draft, title: e.target.value })} />
            </div>
            <Button variant="secondary" size="sm" onClick={() => setPicker("result")} disabled={!active}>
              <Sigma /> Insert a result
            </Button>
            <Button variant="secondary" size="sm" onClick={() => setPicker("citation")} disabled={!active}>
              <Quote /> Insert a citation
            </Button>
          </div>
          <p className="text-xs text-ink-soft">
            Results go in as placeholders such as {"{{mean:metric:condition}}"} (shown as blue chips) and citations as [@key] (green);
            red means unknown. Hover a chip to see what it stands for.
          </p>
          {view.sections.map((section) => (
            <SectionEditor
              key={section.id}
              section={section}
              heading={headings[section.id]}
              value={draft.templates[section.id] ?? section.template}
              problems={checked?.get(section.id) ?? section.problems}
              editor={(handle) => {
                editors.current[section.id] = handle;
              }}
              onChange={(text) => {
                setActive(section.id); // pickers insert into the section edited last
                setDraft((prev) => (prev ? { ...prev, templates: { ...prev.templates, [section.id]: text } } : prev));
              }}
              onFocus={() => setActive(section.id)}
              placeholders={placeholderMap}
              papers={paperTitles}
            />
          ))}
        </div>
        <aside className="space-y-4 xl:sticky xl:top-20">
          <Rail title="Your changes">
            {dirty ? (
              <ul className="space-y-1 text-sm">
                {titleChanged && <li>Title</li>}
                {changed.map((s) => (
                  <li key={s.id}>{headings[s.id]}</li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-ink-soft">Nothing changed yet.</p>
            )}
            <label htmlFor={ids.note} className="block pt-1 text-xs font-medium text-ink-soft">
              Why (optional, kept with the new version)
            </label>
            <Textarea id={ids.note} rows={2} className="min-h-0" value={note} onChange={(e) => setNote(e.target.value)} />
            <div className="flex flex-wrap gap-2 pt-1">
              <Button onClick={apply} busy={decide.isPending} disabled={!dirty || !check.data?.ok || check.isFetching || !sameJson(settled, value)}>
                Apply my edits
              </Button>
              <Button variant="ghost" onClick={discard}>
                Discard
              </Button>
            </div>
            <p className="text-xs text-ink-soft">{savingDraft ? "Saving the draft…" : saved.data?.saved_at && dirty ? `Draft saved ${timeAgo(saved.data.saved_at)}` : ""}</p>
          </Rail>
          <Rail title="Checks">
            {!dirty ? (
              <p className="text-sm text-ink-soft">Change a section to check it.</p>
            ) : check.data && !check.data.ok ? (
              <p className="flex gap-1.5 text-sm text-bad">
                <CircleAlert className="mt-0.5 size-4 shrink-0" /> {check.data.error}
              </p>
            ) : check.data ? (
              <>
                <p className="flex items-center gap-1.5 text-sm text-ok">
                  <CircleCheck className="size-4" /> Placeholders and citations are all known.
                </p>
                {typedNumbers.map((f, i) => (
                  <p key={i} className="flex gap-1.5 text-sm text-warn">
                    <CircleAlert className="mt-0.5 size-4 shrink-0" /> {f.detail} ({headings[f.section]}). The review will flag it; a placeholder keeps it traceable.
                  </p>
                ))}
              </>
            ) : (
              <Spinner label="Checking" />
            )}
          </Rail>
          <Rail title="What applying does">
            <div className="space-y-1 text-sm">
              <p>Edited sections are marked as yours and kept if the writer runs again.</p>
              {which === "final_report" ? (
                <p className="text-ink-soft">The final report is not reviewed again; the checkpoint shows your version.</p>
              ) : redo.length ? (
                <p>Will be redone: {redo.map((a) => stageTitle(a.stage)).join(", ")} (the reviewer reads your version).</p>
              ) : (
                <p className="text-ink-soft">No later stage has started yet.</p>
              )}
            </div>
          </Rail>
        </aside>
      </div>
      <Picker
        open={picker === "result"}
        onClose={() => setPicker(null)}
        title="Insert a result"
        items={view.placeholders.map((p) => ({ key: p.key, label: `${p.text}`, detail: `${p.meaning} · {{${p.key}}}`, insert: `{{${p.key}}}` }))}
        onPick={(text) => active && editors.current[active]?.insert(text)}
      />
      <Picker
        open={picker === "citation"}
        onClose={() => setPicker(null)}
        title="Insert a citation"
        items={view.papers.map((p) => ({ key: p.key, label: p.title, detail: `${p.authors} (${p.year ?? "n.d."}) · [@${p.key}]`, insert: `[@${p.key}]` }))}
        onPick={(text) => active && editors.current[active]?.insert(text)}
      />
    </div>
  );
}
