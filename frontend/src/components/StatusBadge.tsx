import type { RunDetail, StageSummary } from "@/api/types";
import { cn } from "@/lib/cn";
import { stageState, TONE_DOT } from "@/lib/stages";

import { Badge } from "./ui/badge";

export function StatusBadge({ stage, run }: { stage: StageSummary; run?: RunDetail }) {
  const state = stageState(stage, run);
  return (
    <Badge tone={state.tone}>
      <span className={cn("size-1.5 rounded-full", TONE_DOT[state.tone])} aria-hidden />
      {state.label}
    </Badge>
  );
}
