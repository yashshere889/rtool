import { RotateCcw, Undo2 } from "lucide-react";
import { useState } from "react";

import { useDecide, useImpact, useUndo } from "@/api/hooks";
import type { Affected, RunDetail, StageSummary } from "@/api/types";
import { stageTitle } from "@/lib/stages";

import { Button } from "./ui/button";
import { Dialog, DialogClose, DialogContent } from "./ui/dialog";
import { ErrorNote, Loading } from "./ui/states";

function effect(item: Affected): string {
  switch (item.effect) {
    case "redo":
      return item.status === "awaiting_gate" ? "waiting for your decision; would be redone" : "would be redone";
    case "cancel_jobs":
      return `${item.detail} would be cancelled and the stage redone`;
    default:
      return "not started yet; would use the new outcome";
  }
}

/** Open a finished stage's checkpoint again, after showing what a different outcome would mean for later stages. */
export function ReopenDialog({
  run,
  stage,
  open,
  onOpenChange,
  title,
  description,
}: {
  run: RunDetail;
  stage: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: string;
}) {
  const impact = useImpact(run.project, run.run_id, stage, open);
  const decide = useDecide(run.project, run.run_id);
  const affected = impact.data ?? [];
  const started = affected.filter((a) => a.effect !== "not_started");
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent title={title} description={description}>
        <div className="space-y-3 text-sm">
          <p>
            Nothing changes until you decide at the reopened checkpoint. If you keep things as they are, every later
            stage keeps its results.
          </p>
          {impact.isPending && <Loading label="Checking the later stages" />}
          {impact.isError && <ErrorNote error={impact.error} />}
          {impact.data && (
            <div>
              <p className="font-medium">If the outcome changes:</p>
              {started.length === 0 ? (
                <p className="mt-1 text-ink-soft">No later stage has started yet, so nothing would be redone.</p>
              ) : (
                <ul className="mt-1 list-disc space-y-0.5 pl-5">
                  {affected.map((item) => (
                    <li key={item.stage} className={item.effect === "not_started" ? "text-ink-soft" : undefined}>
                      <span className="font-medium">{stageTitle(item.stage)}</span>: {effect(item)}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
        <div className="mt-5 flex justify-end gap-2">
          <DialogClose asChild>
            <Button variant="ghost">Cancel</Button>
          </DialogClose>
          <Button
            busy={decide.isPending}
            onClick={() =>
              decide.mutate(
                { request: { kind: "reopen", stage, payload: {} }, label: `Reopen ${stageTitle(stage)}` },
                { onSuccess: () => onOpenChange(false) },
              )
            }
          >
            Reopen the checkpoint
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/** "Reopen" for a finished stage with a checkpoint; while the request waits for a worker it can be undone. */
export function ReopenButton({
  run,
  stage,
  label,
  title,
  description,
}: {
  run: RunDetail;
  stage: StageSummary;
  label: string;
  title: string;
  description: string;
}) {
  const [open, setOpen] = useState(false);
  const undo = useUndo(run.project, run.run_id);
  const queued = run.queued_actions.find((a) => a.kind === "reopen" && a.stage === stage.id);
  if (queued) {
    return (
      <div className="flex flex-wrap items-center gap-2 text-sm">
        Reopening is on its way.
        <Button variant="secondary" size="sm" busy={undo.isPending} onClick={() => undo.mutate(queued.id)}>
          <Undo2 /> Undo
        </Button>
      </div>
    );
  }
  if (stage.status !== "done" || !stage.checkpoint) return null;
  return (
    <>
      <Button variant="secondary" onClick={() => setOpen(true)}>
        <RotateCcw /> {label}
      </Button>
      {open && (
        <ReopenDialog
          run={run}
          stage={stage.id}
          open={open}
          onOpenChange={setOpen}
          title={title}
          description={description}
        />
      )}
    </>
  );
}
