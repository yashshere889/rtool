import { Check, CircleAlert, Hand, Loader, RotateCcw } from "lucide-react";
import { NavLink } from "react-router";

import { runPath } from "@/api/client";
import type { RunDetail, RunSummary, StageSummary } from "@/api/types";
import { cn } from "@/lib/cn";
import { stageState, stageTitle, TONE_DOT, type Tone } from "@/lib/stages";

const ICONS: Partial<Record<Tone, typeof Check>> = { ok: Check, warn: Hand, bad: CircleAlert, info: Loader, stale: RotateCcw };

export function StageIcon({ stage, run, index }: { stage: StageSummary; run?: RunDetail; index: number }) {
  const state = stageState(stage, run);
  const Icon = ICONS[state.tone];
  return (
    <span
      className={cn(
        "flex size-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold",
        state.tone === "neutral" ? "border border-line bg-surface text-ink-soft" : `${TONE_DOT[state.tone]} text-white`,
      )}
      aria-hidden
    >
      {Icon ? <Icon className={cn("size-3.5", state.tone === "info" && "animate-spin [animation-duration:2.5s]")} /> : index + 1}
    </span>
  );
}

/** The pipeline as a row of steps (overview page). */
export function StageStepper({ run }: { run: RunDetail }) {
  return (
    <ol className="flex flex-wrap gap-2" aria-label="Stages">
      {run.stages.map((stage, i) => {
        const state = stageState(stage, run);
        return (
          <li key={stage.id}>
            <NavLink
              to={`${runPath(run.project, run.run_id)}/${stage.id}`}
              className="flex items-center gap-2 rounded-lg border border-line bg-surface py-1.5 pr-3 pl-1.5 text-sm hover:bg-muted"
            >
              <StageIcon stage={stage} run={run} index={i} />
              <span className="font-medium">{stageTitle(stage.id)}</span>
              <span className="text-ink-soft">· {state.label}</span>
            </NavLink>
          </li>
        );
      })}
    </ol>
  );
}

/** A compact row of dots, one per stage (project list). */
export function StageDots({ run }: { run: RunSummary }) {
  const done = run.stages.filter((s) => s.status === "done").length;
  return (
    <span className="flex items-center gap-1" role="img" aria-label={`${done} of ${run.stages.length} stages done`}>
      {run.stages.map((stage) => {
        const state = stageState(stage);
        return (
          <span
            key={stage.id}
            title={`${stageTitle(stage.id)}: ${state.label}`}
            className={cn("h-1.5 w-5 rounded-full", state.tone === "neutral" ? "bg-line" : TONE_DOT[state.tone])}
          />
        );
      })}
    </span>
  );
}
