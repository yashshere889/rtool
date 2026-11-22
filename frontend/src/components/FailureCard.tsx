import { CircleAlert } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router";

import { runPath } from "@/api/client";
import { useCode, useDecide, useEvents } from "@/api/hooks";
import type { RunDetail, StageSummary } from "@/api/types";
import { repairLabel } from "@/lib/code";
import { lastFailure } from "@/lib/events";
import { dateTime, timeAgo } from "@/lib/format";
import { stageTitle } from "@/lib/stages";

import { Button } from "./ui/button";
import { Textarea } from "./ui/field";

const linkStyle = "text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent";

/** The coding stage: what it repaired on its own, and where to fix experiment.py by hand. */
function CodeHelp({ run, stage }: { run: RunDetail; stage: StageSummary }) {
  const code = useCode(run.project, run.run_id);
  const base = `${runPath(run.project, run.run_id)}/${stage.id}`;
  const fixes = code.data?.fixes ?? [];
  const kinds = [...new Set(fixes.map((f) => repairLabel(f.signature)))];
  return (
    <div className="mt-3 space-y-1 text-sm">
      {fixes.length > 0 && (
        <p>
          It repaired the code {fixes.length} time{fixes.length === 1 ? "" : "s"} on its own first ({kinds.join(", ")}).{" "}
          <Link to={`${base}?tab=tests`} className={linkStyle}>
            See the tests and repairs
          </Link>
        </p>
      )}
      {code.data?.editable && (
        <p>
          You can also fix experiment.py yourself:{" "}
          <Link to={`${base}?tab=code`} className={linkStyle}>
            edit it
          </Link>{" "}
          and choose “Save and resume”. Your version is tested on a CPU first.
        </p>
      )}
    </div>
  );
}

/** A failed stage: what went wrong, what was already tried, and how to try again (optionally with guidance). */
export function FailureCard({ run, stage }: { run: RunDetail; stage: StageSummary }) {
  const decide = useDecide(run.project, run.run_id);
  const events = useEvents(run.project, run.run_id);
  const [note, setNote] = useState("");
  const failure = events.data ? lastFailure(events.data.events, stage.id) : null;
  const retry = (resume: boolean) =>
    decide.mutate({
      request: { kind: "retry", stage: stage.id, payload: { resume, note: note.trim() } },
      label: `${stageTitle(stage.id)}: ${resume ? "continue" : "try again"}`,
    });
  const queued = run.queued_actions.some((a) => a.kind === "retry" && a.stage === stage.id);
  return (
    <section aria-label="Needs attention" className="rounded-xl border border-bad/30 bg-bad-soft/50 p-5">
      <p className="flex flex-wrap items-center gap-x-2 text-xs font-semibold tracking-wide text-bad uppercase">
        <CircleAlert className="size-4" /> Needs attention · {stageTitle(stage.id)}
      </p>
      <p className="mt-2 text-sm font-medium">
        This stage stopped with an error
        {failure && (
          <>
            {" "}
            <time dateTime={failure.ts} title={dateTime(failure.ts)}>
              {timeAgo(failure.ts)}
            </time>{" "}
            and is waiting for you
          </>
        )}
        :
      </p>
      <pre className="mt-2 max-h-48 overflow-auto rounded-lg bg-surface p-3 font-mono text-xs whitespace-pre-wrap">
        {stage.last_error || "no error message was recorded"}
      </pre>
      {failure && failure.retries > 0 && (
        <p className="mt-3 text-sm">
          The tool already tried again {failure.retries} time{failure.retries === 1 ? "" : "s"} by itself before
          stopping.
        </p>
      )}
      {stage.agent === "coder" && <CodeHelp run={run} stage={stage} />}
      {failure?.traceback && (
        <details className="mt-3 text-sm">
          <summary className="cursor-pointer text-ink-soft">Technical details (for whoever maintains rtool)</summary>
          <pre className="mt-2 max-h-64 overflow-auto rounded-lg bg-surface p-3 font-mono text-xs whitespace-pre-wrap">
            {failure.traceback}
          </pre>
        </details>
      )}
      <label className="mt-4 block text-sm font-medium" htmlFor={`note-${stage.id}`}>
        Guidance for the next attempt (optional)
      </label>
      <Textarea
        id={`note-${stage.id}`}
        className="mt-1.5 min-h-16"
        value={note}
        onChange={(e) => setNote(e.target.value)}
        placeholder="For example: if a dataset is gated, use the public mirror instead."
      />
      <div className="mt-3 flex flex-wrap gap-2">
        <Button onClick={() => retry(true)} busy={decide.isPending} disabled={queued}>
          Continue from where it stopped
        </Button>
        <Button variant="secondary" onClick={() => retry(false)} disabled={decide.isPending || queued}>
          Start this stage again
        </Button>
      </div>
      {queued && <p className="mt-2 text-sm text-ink-soft">A retry is queued; it starts when a worker picks it up.</p>}
    </section>
  );
}
