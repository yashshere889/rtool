import { Check, Clock, Loader2 } from "lucide-react";

import { useEvents } from "@/api/hooks";
import type { RunDetail, StageSummary } from "@/api/types";
import { cn } from "@/lib/cn";
import { stageProgress } from "@/lib/events";
import { dateTime, timeAgo } from "@/lib/format";

const STEP_STYLE = {
  done: "border-ok/40 bg-ok-soft/50 text-ok",
  current: "border-info/40 bg-info-soft/60 text-info",
  waiting: "border-line text-ink-soft",
  failed: "border-bad/50 bg-bad-soft/60 text-bad",
  attention: "border-warn/50 bg-warn-soft/60 text-warn",
};

/** A running stage: whether it is still queued, its steps (for stages that report them), and its latest activity. */
export function StageProgress({ run, stage }: { run: RunDetail; stage: StageSummary }) {
  const events = useEvents(run.project, run.run_id);
  if (!events.data) return null;
  const { since, steps, latest, queued } = stageProgress(events.data.events, stage.id, stage.agent);
  return (
    <div className="space-y-3 text-sm">
      {queued && (
        <p className="flex items-start gap-2">
          <Clock className="mt-0.5 size-4 shrink-0 text-ink-soft" />
          <span>
            Waiting in the cluster queue since{" "}
            <time dateTime={queued.since} title={dateTime(queued.since)}>
              {timeAgo(queued.since)}
            </time>
            {queued.job && ` (job ${queued.job})`}
            {queued.expectedStart && `; the cluster expects it to start ${queued.expectedStart.replace("T", " ")}`}.
          </span>
        </p>
      )}
      {steps && (
        <ol aria-label={`Steps of ${stage.id}`} className="flex flex-wrap gap-1.5">
          {steps.map((step, i) => (
            <li
              key={step.label}
              aria-current={step.state === "current" ? "step" : undefined}
              className={cn("flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs", STEP_STYLE[step.state])}
            >
              {step.state === "done" ? (
                <Check className="size-3.5" />
              ) : step.state === "current" ? (
                <Loader2 className="size-3.5 animate-spin motion-reduce:animate-none" />
              ) : (
                <span className="w-3.5 text-center">{i + 1}</span>
              )}
              {step.label}
            </li>
          ))}
        </ol>
      )}
      {(since || latest) && (
        <p className="text-ink-soft">
          {since && (
            <>
              Started{" "}
              <time dateTime={since} title={dateTime(since)}>
                {timeAgo(since)}
              </time>
            </>
          )}
          {since && latest && " · "}
          {latest && (
            <>
              latest: {latest.text} ({timeAgo(latest.ts)})
            </>
          )}
        </p>
      )}
    </div>
  );
}
