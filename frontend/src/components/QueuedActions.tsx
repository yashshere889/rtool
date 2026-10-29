import { Undo2 } from "lucide-react";

import { useUndo } from "@/api/hooks";
import type { RunDetail } from "@/api/types";
import { timeAgo } from "@/lib/format";
import { gateInfo, stageTitle } from "@/lib/stages";

import { Button } from "./ui/button";

/** Decisions sent but not yet applied by a worker; they can still be taken back. */
export function QueuedActions({ run }: { run: RunDetail }) {
  const undo = useUndo(run.project, run.run_id);
  if (!run.queued_actions.length) return null;
  return (
    <ul className="space-y-2">
      {run.queued_actions.map((action) => (
        <li key={action.id} className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-muted px-3 py-2 text-sm">
          <span>
            <span className="font-medium capitalize">{action.kind}</span>
            {action.gate ? ` · ${gateInfo(action.gate).title}` : action.stage ? ` · ${stageTitle(action.stage)}` : ""}
            <span className="text-ink-soft"> · sent {timeAgo(action.created_at)}, waiting for a worker</span>
          </span>
          <Button variant="ghost" size="sm" busy={undo.isPending} onClick={() => undo.mutate(action.id)}>
            <Undo2 /> Undo
          </Button>
        </li>
      ))}
    </ul>
  );
}
