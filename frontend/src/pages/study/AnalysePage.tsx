import { FileText } from "lucide-react";

import { fileUrl } from "@/api/client";
import { useArtifact } from "@/api/hooks";
import type { RunDetail, StageSummary } from "@/api/types";
import { FailureCard } from "@/components/FailureCard";
import { StatusBadge } from "@/components/StatusBadge";
import { Empty } from "@/components/ui/states";
import { STAGES } from "@/lib/stages";

import { ResultsTab } from "../code/ResultsTab";

/** The study's analysis: rtool's own script over the uploaded table, and what it found. */
export function AnalysePage({ run, stage }: { run: RunDetail; stage: StageSummary }) {
  const bundle = useArtifact(run.project, run.run_id, "code_bundle", undefined, stage.status === "done");
  const root = (bundle.data?.payload as { root?: string } | undefined)?.root;
  return (
    <div className="space-y-6">
      <header>
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-semibold">Analysis</h1>
          <StatusBadge stage={stage} run={run} />
          {root && (
            <a className="ml-auto inline-flex items-center gap-1 text-sm text-accent underline underline-offset-2" href={fileUrl(run.project, run.run_id, `${root}/experiment.py`)} target="_blank" rel="noreferrer">
              <FileText className="size-4" /> The analysis script
            </a>
          )}
        </div>
        <p className="mt-2 max-w-3xl text-sm text-ink-soft">{STAGES.analyse.description}</p>
      </header>
      {stage.status === "failed" && <FailureCard run={run} stage={stage} />}
      {stage.status === "done" ? (
        <ResultsTab run={run} />
      ) : (
        stage.status !== "failed" && <Empty title="Nothing to analyse yet">The analysis runs once the study's data are uploaded.</Empty>
      )}
    </div>
  );
}
