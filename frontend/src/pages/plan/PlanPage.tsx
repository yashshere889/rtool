import { useQueryClient } from "@tanstack/react-query";
import { CircleAlert, CircleCheck, FileText, Pencil, Undo2 } from "lucide-react";
import { useEffect, useId, useMemo, useState, type ReactNode } from "react";
import { useSearchParams } from "react-router";

import { ApiError, fileUrl, runPath } from "@/api/client";
import { keys, useDecide, useDiscardDraft, useDraft, useImpact, usePlan, usePlanCheck, useSaveDraft, useUndo } from "@/api/hooks";
import type { ExperimentPlan, PlanView, RunDetail, StageSummary } from "@/api/types";
import { CheckpointCard } from "@/components/CheckpointCard";
import { FailureCard } from "@/components/FailureCard";
import { StatusBadge } from "@/components/StatusBadge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/field";
import { Spinner } from "@/components/ui/spinner";
import { Empty, ErrorNote, Loading } from "@/components/ui/states";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { sameJson } from "@/lib/equal";
import { dateTime, timeAgo } from "@/lib/format";
import { planChanges } from "@/lib/plan";
import { STAGES, stageTitle } from "@/lib/stages";
import { useDebounced } from "@/lib/useDebounced";

import { DownloadsPanel } from "./DownloadsPanel";
import { PlanForm } from "./PlanForm";
import { PlanRead } from "./PlanRead";

const ARTIFACT = "experiment_plan";

function Rail({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-2 rounded-xl border border-line bg-surface p-4">
      <h2 className="text-sm font-semibold">{title}</h2>
      {children}
    </section>
  );
}

function Issues({ issues }: { issues: string[] }) {
  if (!issues.length) {
    return (
      <p className="flex items-center gap-1.5 text-sm text-ok">
        <CircleCheck className="size-4" /> No problems found.
      </p>
    );
  }
  return (
    <ul className="space-y-1.5 text-sm">
      {issues.map((issue) => (
        <li key={issue} className="flex gap-1.5 text-warn">
          <CircleAlert className="mt-0.5 size-4 shrink-0" /> <span>{issue}</span>
        </li>
      ))}
    </ul>
  );
}

/** The plan as JSON, for those who prefer it; only valid JSON is taken over. */
function JsonEditor({ value, onChange }: { value: ExperimentPlan; onChange: (plan: ExperimentPlan) => void }) {
  const [text, setText] = useState(() => JSON.stringify(value, null, 2));
  const [error, setError] = useState("");
  const id = useId();
  return (
    <div className="space-y-2">
      <label htmlFor={id} className="text-sm text-ink-soft">
        The whole plan as JSON. Derived parts (runs, GPU-hours, results schema, issues) are worked out again.
      </label>
      <Textarea
        id={id}
        spellCheck={false}
        className="min-h-[60vh] font-mono text-xs"
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          try {
            const parsed = JSON.parse(e.target.value);
            if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) throw new Error("the plan must be an object");
            setError("");
            onChange(parsed as ExperimentPlan);
          } catch (exc) {
            setError((exc as Error).message);
          }
        }}
      />
      {error && <p className="text-sm text-bad">Not valid JSON yet: {error}</p>}
    </div>
  );
}

function ImpactNote({ run, stage }: { run: RunDetail; stage: StageSummary }) {
  const impact = useImpact(run.project, run.run_id, stage.id);
  const redo = (impact.data ?? []).filter((a) => a.effect !== "not_started");
  if (impact.isPending) return <Spinner label="Checking later stages" />;
  return (
    <div className="space-y-1 text-sm">
      {stage.status === "awaiting_gate" && <p>The checkpoint stays open and shows your version for approval.</p>}
      {redo.length ? (
        <p>
          Will be redone:{" "}
          {redo.map((a) => `${stageTitle(a.stage)}${a.effect === "cancel_jobs" ? ` (${a.detail} cancelled)` : ""}`).join(", ")}.
        </p>
      ) : (
        <p className="text-ink-soft">No later stage has started, so nothing is redone.</p>
      )}
    </div>
  );
}

function VersionNote({ view }: { view: PlanView }) {
  return (
    <p className="text-sm text-ink-soft" title={dateTime(view.created_at)}>
      Version {view.version}
      {view.by ? ` · edited by ${view.by.replace(/ \(web\)$/, "")} from version ${view.edited_from}` : " · written by the planner"} ·{" "}
      {timeAgo(view.created_at)}
      {view.edit_note && <span className="block">“{view.edit_note}”</span>}
    </p>
  );
}

export function PlanPage({ run, stage }: { run: RunDetail; stage: StageSummary }) {
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") ?? "read";
  const versionParam = params.get("version");
  const plan = usePlan(run.project, run.run_id, versionParam ? Number(versionParam) : undefined);
  const saved = useDraft(run.project, run.run_id, ARTIFACT);
  const saveDraft = useSaveDraft(run.project, run.run_id, ARTIFACT);
  const discardDraft = useDiscardDraft(run.project, run.run_id, ARTIFACT);
  const decide = useDecide(run.project, run.run_id);
  const undo = useUndo(run.project, run.run_id);
  const queryClient = useQueryClient();
  const [draft, setDraft] = useState<ExperimentPlan | null>(null);
  const [note, setNote] = useState("");
  const noteId = useId();

  const view = plan.data;
  const queuedEdit = run.queued_actions.find((a) => a.kind === "edit" && a.payload.artifact === ARTIFACT);
  const editing = (tab === "edit" || tab === "json") && Boolean(view?.editable) && !queuedEdit;
  const staleDraft = saved.data && view && saved.data.base_version !== view.head ? saved.data : null;

  const setParam = (key: string, value: string | null) =>
    setParams((prev) => {
      const next = new URLSearchParams(prev);
      if (value === null) next.delete(key);
      else next.set(key, value);
      return next;
    });

  // start editing from the saved draft (if it was made on the latest version) or from the plan
  useEffect(() => {
    if (!editing || draft || !view || saved.isPending) return;
    if (saved.data && saved.data.base_version === view.head) {
      setDraft(saved.data.value as unknown as ExperimentPlan);
      setNote(saved.data.note);
    } else {
      setDraft(structuredClone(view.plan));
    }
  }, [editing, draft, view, saved.isPending, saved.data]);

  const dirty = Boolean(draft && view && !sameJson(draft, view.plan));
  const settled = useDebounced(draft, 600);
  const check = usePlanCheck(run.project, run.run_id, view?.head ?? 0, settled, Boolean(editing && settled && dirty));
  const changes = useMemo(() => (draft && view ? planChanges(view.plan, draft) : []), [draft, view]);

  // keep the draft on the server as it is typed, so a reload or another device loses nothing
  const pending = useMemo(
    () => (editing && dirty && view ? { base_version: view.head, value: draft, note } : null),
    [editing, dirty, view, draft, note],
  );
  const toSave = useDebounced(pending, 1000);
  const { mutate: save } = saveDraft;
  useEffect(() => {
    if (toSave) save(toSave as never);
  }, [toSave, save]);

  const header = (
    <header>
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold">Experiment plan</h1>
        <StatusBadge stage={stage} run={run} />
      </div>
      <p className="mt-2 max-w-3xl text-sm text-ink-soft">{STAGES.plan.description}</p>
    </header>
  );

  if (plan.isPending) return <div className="space-y-6">{header}<Loading label="Loading the plan" /></div>;
  if (plan.isError || !view) {
    const missing = plan.error instanceof ApiError && plan.error.status === 404;
    return (
      <div className="space-y-6">
        {header}
        {stage.status === "failed" && <FailureCard run={run} stage={stage} />}
        {missing ? (
          <Empty title="No plan yet">It appears here once you have chosen a hypothesis and the planner has designed the experiment.</Empty>
        ) : (
          <ErrorNote error={plan.error} />
        )}
      </div>
    );
  }

  const latest = view.version === view.head;
  const shownIssues = editing && dirty ? (check.data?.ok ? (check.data.plan?.issues ?? []) : null) : view.plan.issues;

  function apply() {
    if (!draft || !view) return;
    decide.mutate(
      {
        request: { kind: "edit", stage: stage.id, payload: { artifact: ARTIFACT, base_version: view.head, value: draft, note: note.trim() } },
        label: "Your plan edits",
      },
      {
        onSuccess: () => {
          setDraft(null);
          setNote("");
          queryClient.setQueryData(keys.draft(run.project, run.run_id, ARTIFACT), null); // the queued edit holds it now
          setParam("tab", null);
        },
      },
    );
  }

  function discard() {
    discardDraft.mutate();
    setDraft(null);
    setNote("");
    setParam("tab", null);
  }

  return (
    <div className="space-y-6">
      {header}
      {stage.gate && (
        <CheckpointCard
          run={run}
          gate={stage.gate}
          showStageLink={false}
          rejectHint={view.by ? "Your hand edits are not kept: the planner writes a new plan from your comments." : undefined}
        />
      )}
      {stage.status === "failed" && <FailureCard run={run} stage={stage} />}
      {stage.status === "stale" && stage.stale_reason && (
        <p className="rounded-lg bg-stale-soft px-4 py-3 text-sm text-stale">Will run again: {stage.stale_reason}</p>
      )}
      {queuedEdit && (
        <div className="flex flex-wrap items-center gap-3 rounded-lg bg-info-soft px-4 py-3 text-sm text-info">
          Your edits to the plan are on their way; they are applied when a worker picks them up.
          <Button variant="secondary" size="sm" busy={undo.isPending} onClick={() => undo.mutate(queuedEdit.id)}>
            <Undo2 /> Undo and keep editing
          </Button>
        </div>
      )}
      {staleDraft && latest && !queuedEdit && (
        <div className="flex flex-wrap items-center gap-3 rounded-lg bg-warn-soft px-4 py-3 text-sm text-warn">
          You have unsent edits made on version {staleDraft.base_version}, but the plan has changed since (now version {view.head}).
          <Button variant="secondary" size="sm" onClick={() => discardDraft.mutate()}>
            Discard them
          </Button>
        </div>
      )}
      <DownloadsPanel run={run} />

      <div className="flex flex-wrap items-center gap-3">
        <VersionNote view={view} />
        <div className="ml-auto flex flex-wrap items-center gap-2">
          {view.plan.markdown_path && (
            <a href={fileUrl(run.project, run.run_id, view.plan.markdown_path)} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-sm text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent">
              <FileText className="size-4" /> Plan as a document
            </a>
          )}
          {view.versions.length > 1 && (
            <select
              aria-label="Plan version"
              className="h-8 rounded-md border border-line bg-surface px-2 text-sm"
              value={view.version}
              onChange={(e) => {
                const value = Number(e.target.value);
                setParam("version", value === view.head ? null : String(value));
                setParam("tab", null);
              }}
            >
              {[...view.versions].reverse().map((v) => (
                <option key={v} value={v}>
                  Version {v}
                  {v === view.head ? " (latest)" : ""}
                </option>
              ))}
            </select>
          )}
        </div>
      </div>

      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_20rem]">
        <Tabs value={editing ? tab : "read"} onValueChange={(value) => setParam("tab", value === "read" ? null : value)}>
          <TabsList>
            <TabsTrigger value="read">Plan</TabsTrigger>
            {view.editable && !queuedEdit && <TabsTrigger value="edit">Edit</TabsTrigger>}
            {view.editable && !queuedEdit && <TabsTrigger value="json">JSON</TabsTrigger>}
          </TabsList>
          <TabsContent value="read" className="mt-4">
            {dirty && (
              <p className="mb-4 rounded-lg bg-info-soft px-4 py-3 text-sm text-info">
                You have unsent edits.{" "}
                <button type="button" className="font-medium underline" onClick={() => setParam("tab", "edit")}>
                  Continue editing
                </button>
              </p>
            )}
            <PlanRead plan={view.plan} cluster={view.cluster} title={view.hypothesis_title} />
          </TabsContent>
          <TabsContent value="edit" className="mt-4">
            {draft && (
              <PlanForm
                value={draft}
                base={view.plan}
                onChange={setDraft}
                cluster={view.cluster}
                checked={check.data}
                title={view.hypothesis_title}
                hypothesesPath={`${runPath(run.project, run.run_id)}/${run.stages.find((s) => s.agent === "hypothesis")?.id ?? "hypotheses"}`}
              />
            )}
          </TabsContent>
          <TabsContent value="json" className="mt-4">
            {draft && tab === "json" && <JsonEditor value={draft} onChange={setDraft} />}
          </TabsContent>
        </Tabs>

        <aside className="space-y-4 xl:sticky xl:top-20">
          {editing ? (
            <>
              <Rail title="Your changes">
                {changes.length ? (
                  <ul className="space-y-1 text-sm">
                    {changes.map((c) => (
                      <li key={c.label}>
                        <span className="font-medium">{c.label}</span>: {c.detail}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-ink-soft">Nothing changed yet.</p>
                )}
                <label htmlFor={noteId} className="block pt-1 text-xs font-medium text-ink-soft">
                  Why (optional, kept with the new version)
                </label>
                <Textarea id={noteId} rows={2} className="min-h-0" value={note} onChange={(e) => setNote(e.target.value)} />
                <div className="flex flex-wrap gap-2 pt-1">
                  <Button
                    onClick={apply}
                    busy={decide.isPending}
                    disabled={!dirty || !check.data?.ok || check.isFetching || settled !== draft}
                  >
                    Apply my edits
                  </Button>
                  <Button variant="ghost" onClick={discard}>
                    Discard
                  </Button>
                </div>
                <p className="text-xs text-ink-soft">
                  {saveDraft.isPending ? "Saving the draft…" : saved.data?.saved_at && dirty ? `Draft saved ${timeAgo(saved.data.saved_at)}` : ""}
                </p>
              </Rail>
              <Rail title="Checks">
                {check.data && !check.data.ok ? (
                  <p className="flex gap-1.5 text-sm text-bad">
                    <CircleAlert className="mt-0.5 size-4 shrink-0" /> {check.data.error}
                  </p>
                ) : shownIssues ? (
                  <Issues issues={shownIssues} />
                ) : (
                  <Spinner label="Checking" />
                )}
              </Rail>
              <Rail title="What applying does">
                <ImpactNote run={run} stage={stage} />
              </Rail>
            </>
          ) : (
            <>
              <Rail title="Checks">
                <Issues issues={view.plan.issues} />
              </Rail>
              {view.editable && !queuedEdit ? (
                <Button className="w-full" variant="secondary" onClick={() => setParam("tab", "edit")}>
                  <Pencil /> Edit the plan
                </Button>
              ) : (
                !queuedEdit && view.why_not_editable && <p className="text-sm text-ink-soft">Editing: {view.why_not_editable}.</p>
              )}
              {view.by && <Badge tone="info">This version was edited by hand</Badge>}
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
