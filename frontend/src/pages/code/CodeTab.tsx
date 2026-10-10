import { CircleAlert, CircleCheck, Download, FileCode2, GitCompare, Pencil, Undo2 } from "lucide-react";
import { lazy, Suspense, useEffect, useState } from "react";

import { useCodeCheck, useCodeFile, useDecide, useDiscardDraft, useDraft, useGeneratedCode, useSaveDraft, useUndo } from "@/api/hooks";
import type { CodeView, RunDetail } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button, buttonStyles } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { ErrorNote, Loading } from "@/components/ui/states";
import { cn } from "@/lib/cn";
import { fileDraftKey } from "@/lib/code";
import { timeAgo } from "@/lib/format";
import { useDebounced } from "@/lib/useDebounced";

const CodeEditor = lazy(() => import("@/components/code/CodeEditor"));
const CodeDiff = lazy(() => import("@/components/code/CodeEditor").then((m) => ({ default: m.CodeDiff })));

const EXPERIMENT = "experiment.py";
const PATH = `code/${EXPERIMENT}`; // as the stage's workspace knows it
const ROLE_LABEL = { experiment: "the experiment", harness: "rtool's harness", config: "settings from the plan", readme: "notes", other: "" };

const editorFallback = <Loading label="Opening the editor" />;

function ReadOnlyFile({ run, path }: { run: RunDetail; path: string }) {
  const file = useCodeFile(run.project, run.run_id, path);
  return (
    <div className="space-y-2">
      <p className="rounded-lg bg-info-soft px-3 py-2 text-sm text-info">
        rtool writes this file and rewrites it on every attempt, so it cannot be edited here. The experiment itself is in
        experiment.py.
      </p>
      {file.isPending && <Loading />}
      {file.isError && <ErrorNote error={file.error} />}
      {file.data &&
        (path.endsWith(".py") ? (
          <Suspense fallback={editorFallback}>
            <CodeEditor value={file.data.content} readOnly label={path} minHeight="12rem" />
          </Suspense>
        ) : (
          <pre className="max-h-[70vh] overflow-auto rounded-xl border border-line bg-surface p-3 font-mono text-xs whitespace-pre-wrap">
            {file.data.content}
          </pre>
        ))}
    </div>
  );
}

function ExperimentEditor({ run, view }: { run: RunDetail; view: CodeView }) {
  const file = useCodeFile(run.project, run.run_id, EXPERIMENT);
  const key = fileDraftKey(PATH);
  const saved = useDraft(run.project, run.run_id, key);
  const { mutate: saveDraft, isPending: savingDraft } = useSaveDraft(run.project, run.run_id, key);
  const discardDraft = useDiscardDraft(run.project, run.run_id, key);
  const decide = useDecide(run.project, run.run_id);
  const undo = useUndo(run.project, run.run_id);
  const [mode, setMode] = useState<"edit" | "diff">("edit");
  const [text, setText] = useState<string | null>(null);
  const [base, setBase] = useState<{ content: string; sha: string } | null>(null); // what the edits started from
  const generated = useGeneratedCode(run.project, run.run_id, Boolean(view.generated_sha));
  const queued = run.queued_actions.find((a) => a.kind === "write_file" && a.stage === view.stage);
  const canEdit = view.editable && !queued;

  // start from the saved draft (if it was made on this version of the file), else from the file
  useEffect(() => {
    if (!file.data || saved.isPending || base) return;
    const draft = saved.data?.value as { content?: string; base_sha?: string } | undefined;
    if (draft?.content !== undefined && draft.base_sha === file.data.sha) setText(draft.content);
    else setText(file.data.content);
    setBase({ content: file.data.content, sha: file.data.sha });
  }, [file.data, saved.isPending, saved.data, base]);

  // the file changed underneath (applied edit, or the agent rewrote it): follow it unless there are local edits
  useEffect(() => {
    if (!file.data || !base || file.data.sha === base.sha) return;
    if (text === base.content || text === file.data.content) {
      setText(file.data.content);
      setBase({ content: file.data.content, sha: file.data.sha });
    }
  }, [file.data, base, text]);

  const dirty = text !== null && base !== null && text !== base.content;
  const stale = Boolean(file.data && base && file.data.sha !== base.sha && dirty);
  const settled = useDebounced(text, 700);
  const check = useCodeCheck(run.project, run.run_id, canEdit && settled !== null ? settled : null);
  const toSave = useDebounced(canEdit && dirty && base ? { base_version: 0, value: { content: text, base_sha: base.sha }, note: "" } : null, 1200);
  useEffect(() => {
    if (toSave) saveDraft(toSave);
  }, [toSave, saveDraft]);

  const atGate = view.status === "awaiting_gate";
  const untested = atGate && view.tested_sha !== null && view.experiment_sha !== view.tested_sha;

  function send(retest: boolean) {
    if (text === null || !base) return;
    decide.mutate({
      request: { kind: "write_file", stage: view.stage, payload: { path: PATH, content: text, base_sha: base.sha, retest } },
      label: retest ? (atGate ? "Save and test again" : "Save and resume") : "Save experiment.py",
    });
  }

  if (file.isPending || text === null) return <Loading label="Opening experiment.py" />;
  if (file.isError) return <ErrorNote error={file.error} />;
  const problems = check.data?.problems ?? [];

  return (
    <div className="space-y-3">
      {queued ? (
        <div className="flex flex-wrap items-center gap-3 rounded-lg bg-info-soft px-3 py-2 text-sm text-info">
          Your changes to experiment.py are on their way.
          <Button variant="secondary" size="sm" busy={undo.isPending} onClick={() => undo.mutate(queued.id)}>
            <Undo2 /> Undo and keep editing
          </Button>
        </div>
      ) : !view.editable ? (
        <p className="rounded-lg bg-muted px-3 py-2 text-sm text-ink-soft">You can read the code; editing is possible later: {view.why_not_editable}.</p>
      ) : (
        <p className="rounded-lg bg-muted px-3 py-2 text-sm text-ink-soft">
          <Pencil className="mr-1 inline size-3.5" />
          {atGate
            ? "You can change experiment.py before approving GPU time. A changed file is tested on a CPU again before anything runs on a GPU."
            : "Fix experiment.py, then save and resume: the stage tests your version first."}
        </p>
      )}
      {untested && !dirty && (
        <p className="rounded-lg bg-warn-soft px-3 py-2 text-sm text-warn">
          experiment.py has changed since its last test. Approving tests it again first and then asks you once more.
        </p>
      )}
      {stale && (
        <p className="rounded-lg bg-warn-soft px-3 py-2 text-sm text-warn">
          The file changed since you started editing. Copy what you need, then discard your edits to see the new version.
        </p>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <div className="inline-flex rounded-lg border border-line bg-surface p-0.5" role="group" aria-label="View">
          <button
            type="button"
            aria-pressed={mode === "edit"}
            onClick={() => setMode("edit")}
            className={cn("inline-flex items-center gap-1.5 rounded-md px-3 py-1 text-sm", mode === "edit" ? "bg-muted font-medium" : "text-ink-soft")}
          >
            <FileCode2 className="size-4" /> {canEdit ? "Edit" : "Code"}
          </button>
          <button
            type="button"
            aria-pressed={mode === "diff"}
            onClick={() => setMode("diff")}
            disabled={!view.generated_sha}
            title={view.generated_sha ? undefined : "This run has no copy of the generated code"}
            className={cn("inline-flex items-center gap-1.5 rounded-md px-3 py-1 text-sm disabled:opacity-50", mode === "diff" ? "bg-muted font-medium" : "text-ink-soft")}
          >
            <GitCompare className="size-4" /> Changes since the tool wrote it
          </button>
        </div>
        {view.experiment_sha !== view.generated_sha && view.generated_sha && <Badge tone="info">edited by hand</Badge>}
        <span className="ml-auto text-xs text-ink-soft">
          {savingDraft ? "Saving the draft…" : dirty && saved.data?.saved_at ? `Draft saved ${timeAgo(saved.data.saved_at)}` : ""}
        </span>
      </div>

      <Suspense fallback={editorFallback}>
        {mode === "diff" ? (
          generated.data ? (
            <CodeDiff key={generated.data.sha} original={generated.data.content} value={text} label="experiment.py compared with the generated version" />
          ) : (
            <Loading label="Loading the generated version" />
          )
        ) : (
          <CodeEditor value={text} onChange={canEdit ? setText : undefined} readOnly={!canEdit} label="experiment.py" />
        )}
      </Suspense>

      {canEdit && (
        <>
          <div className="text-sm" aria-live="polite">
            {check.isFetching && !check.data ? (
              <Spinner label="Checking" />
            ) : problems.length ? (
              <ul className="space-y-1 text-bad">
                {problems.map((p) => (
                  <li key={p} className="flex gap-1.5">
                    <CircleAlert className="mt-0.5 size-4 shrink-0" /> {p}
                  </li>
                ))}
              </ul>
            ) : check.data ? (
              <p className="flex items-center gap-1.5 text-ok">
                <CircleCheck className="size-4" /> The checks the coding agent runs before a test pass.
              </p>
            ) : null}
          </div>
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => send(true)} busy={decide.isPending} disabled={!dirty || problems.length > 0 || stale}>
              {atGate ? "Save and test again" : "Save and resume"}
            </Button>
            <Button variant="secondary" onClick={() => send(false)} disabled={!dirty || decide.isPending || stale}>
              Save only
            </Button>
            {untested && !dirty && (
              <Button
                variant="secondary"
                onClick={() =>
                  decide.mutate({
                    request: { kind: "reject", stage: view.stage, gate: "gpu_runs", payload: { retest: true } },
                    label: "Test experiment.py again",
                  })
                }
              >
                Test it again
              </Button>
            )}
            {dirty && (
              <Button
                variant="ghost"
                onClick={() => {
                  discardDraft.mutate();
                  if (file.data) {
                    setText(file.data.content);
                    setBase({ content: file.data.content, sha: file.data.sha });
                  }
                }}
              >
                Discard my edits
              </Button>
            )}
          </div>
        </>
      )}
    </div>
  );
}

/** The experiment code: every file, with experiment.py editable while the stage allows it. */
export function CodeTab({ run, view }: { run: RunDetail; view: CodeView }) {
  const [selected, setSelected] = useState(EXPERIMENT);
  if (!view.files.length) return <p className="text-sm text-ink-soft">The code has not been written yet.</p>;
  return (
    <div className="grid items-start gap-4 lg:grid-cols-[14rem_minmax(0,1fr)]">
      <nav aria-label="Code files" className="space-y-1">
        {view.files.map((f) => (
          <button
            key={f.path}
            type="button"
            aria-current={selected === f.path ? "true" : undefined}
            onClick={() => setSelected(f.path)}
            className={cn(
              "block w-full rounded-lg px-3 py-2 text-left text-sm",
              selected === f.path ? "bg-accent-soft font-medium" : "hover:bg-muted",
            )}
          >
            <span className="block font-mono text-xs">{f.path}</span>
            <span className="text-xs text-ink-soft">
              {ROLE_LABEL[f.role]}
              {f.lines ? ` · ${f.lines} lines` : ""}
              {f.editable ? " · editable" : ""}
            </span>
          </button>
        ))}
        <a href={`/api/runs/${encodeURIComponent(run.project)}/${encodeURIComponent(run.run_id)}/code/archive`} className={cn(buttonStyles({ variant: "ghost", size: "sm" }), "mt-2 w-full justify-start")}>
          <Download /> Download the code (.zip)
        </a>
      </nav>
      <div className="min-w-0">
        {selected === EXPERIMENT ? <ExperimentEditor run={run} view={view} /> : <ReadOnlyFile run={run} path={selected} />}
      </div>
    </div>
  );
}
