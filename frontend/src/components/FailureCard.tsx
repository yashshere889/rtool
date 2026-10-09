import { CircleAlert } from "lucide-react";
import { useState } from "react";

import { useDecide } from "@/api/hooks";
import type { RunDetail, StageSummary } from "@/api/types";
import { stageTitle } from "@/lib/stages";

import { Button } from "./ui/button";
import { Textarea } from "./ui/field";

/** A failed stage: what went wrong, and how to try again (optionally with guidance for the agent). */
export function FailureCard({ run, stage }: { run: RunDetail; stage: StageSummary }) {
  const decide = useDecide(run.project, run.run_id);
  const [note, setNote] = useState("");
  const retry = (resume: boolean) =>
    decide.mutate({
      request: { kind: "retry", stage: stage.id, payload: { resume, note: note.trim() } },
      label: `${stageTitle(stage.id)}: ${resume ? "continue" : "try again"}`,
    });
  const queued = run.queued_actions.some((a) => a.kind === "retry" && a.stage === stage.id);
  return (
    <section aria-label="Needs attention" className="rounded-xl border border-bad/30 bg-bad-soft/50 p-5">
      <p className="flex items-center gap-2 text-xs font-semibold tracking-wide text-bad uppercase">
        <CircleAlert className="size-4" /> Needs attention · {stageTitle(stage.id)}
      </p>
      <p className="mt-2 text-sm font-medium">This stage stopped with an error:</p>
      <pre className="mt-2 max-h-48 overflow-auto rounded-lg bg-surface p-3 font-mono text-xs whitespace-pre-wrap">
        {stage.last_error || "no error message was recorded"}
      </pre>
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
