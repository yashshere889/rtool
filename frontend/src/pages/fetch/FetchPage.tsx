import { useDownloads } from "@/api/hooks";
import type { RunDetail, StageSummary } from "@/api/types";
import { CheckpointCard } from "@/components/CheckpointCard";
import { FailureCard } from "@/components/FailureCard";
import { StatusBadge } from "@/components/StatusBadge";
import { Empty } from "@/components/ui/states";
import { STAGES } from "@/lib/stages";

import { DownloadsPanel } from "../plan/DownloadsPanel";

/** The datasets and models of the approved plan, how each came (or why not), and the manual-resources checkpoint. */
export function FetchPage({ run, stage }: { run: RunDetail; stage: StageSummary }) {
  const downloads = useDownloads(run.project, run.run_id);
  return (
    <div className="space-y-6">
      <header>
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-semibold">Datasets & models</h1>
          <StatusBadge stage={stage} run={run} />
        </div>
        <p className="mt-2 max-w-3xl text-sm text-ink-soft">{STAGES.fetch.description}</p>
      </header>
      {stage.gate && <CheckpointCard run={run} gate={stage.gate} showStageLink={false} />}
      {stage.status === "failed" && <FailureCard run={run} stage={stage} />}
      {downloads.data?.started ? (
        <DownloadsPanel run={run} />
      ) : (
        <Empty title="Nothing downloaded yet">The downloads start once you approve the experiment plan.</Empty>
      )}
    </div>
  );
}
