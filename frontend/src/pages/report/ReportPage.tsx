import type { RunDetail, StageSummary } from "@/api/types";
import { CheckpointCard } from "@/components/CheckpointCard";
import { FailureCard } from "@/components/FailureCard";
import { StatusBadge } from "@/components/StatusBadge";
import { STAGES } from "@/lib/stages";

import { ReportWorkspace } from "./ReportWorkspace";

export function ReportPage({ run, stage }: { run: RunDetail; stage: StageSummary }) {
  return (
    <div className="space-y-6">
      <header className="print:hidden">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-semibold">Report</h1>
          <StatusBadge stage={stage} run={run} />
        </div>
        <p className="mt-2 max-w-3xl text-sm text-ink-soft">{STAGES.report.description}</p>
      </header>
      {stage.gate && <CheckpointCard run={run} gate={stage.gate} showStageLink={false} />}
      {stage.status === "failed" && <FailureCard run={run} stage={stage} />}
      {stage.status === "stale" && stage.stale_reason && (
        <p className="rounded-lg bg-stale-soft px-4 py-3 text-sm text-stale print:hidden">Will run again: {stage.stale_reason}</p>
      )}
      <ReportWorkspace run={run} which="report" stage={stage.id} />
    </div>
  );
}
