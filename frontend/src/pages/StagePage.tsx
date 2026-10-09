import { useState } from "react";
import { useParams } from "react-router";

import { useArtifact, useArtifacts } from "@/api/hooks";
import type { RunDetail } from "@/api/types";
import { ActivityFeed } from "@/components/ActivityFeed";
import { CheckpointCard } from "@/components/CheckpointCard";
import { FailureCard } from "@/components/FailureCard";
import { JsonView } from "@/components/JsonView";
import { useRunContext } from "@/components/layout/RunLayout";
import { NoteBox } from "@/components/NoteBox";
import { ReopenButton } from "@/components/ReopenDialog";
import { StatusBadge } from "@/components/StatusBadge";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { HypothesesPage } from "@/pages/hypotheses/HypothesesPage";
import { LiteraturePage } from "@/pages/literature/LiteraturePage";
import { Empty, ErrorNote, Loading } from "@/components/ui/states";
import { dateTime, humanize, timeAgo } from "@/lib/format";
import { STAGES, stageTitle } from "@/lib/stages";

function ArtifactPanel({ run, name, current }: { run: RunDetail; name: string; current: number }) {
  const artifacts = useArtifacts(run.project, run.run_id);
  const versions = artifacts.data?.find((a) => a.name === name)?.versions ?? [current];
  const [version, setVersion] = useState(current);
  const envelope = useArtifact(run.project, run.run_id, name, version);
  return (
    <Card>
      <CardHeader
        title={humanize(name)}
        description={
          envelope.data
            ? `Version ${envelope.data.version} · by ${envelope.data.producer.agent} · ${timeAgo(envelope.data.created_at)}`
            : undefined
        }
        action={
          versions.length > 1 && (
            <label className="flex items-center gap-2 text-sm">
              <span className="text-ink-soft">Version</span>
              <select
                className="h-8 rounded-md border border-line bg-surface px-2"
                value={version}
                onChange={(e) => setVersion(Number(e.target.value))}
              >
                {[...versions].reverse().map((v) => (
                  <option key={v} value={v}>
                    {v}
                    {v === current ? " (current)" : ""}
                  </option>
                ))}
              </select>
            </label>
          )
        }
      />
      <CardBody>
        {envelope.isPending && <Loading />}
        {envelope.isError && <ErrorNote error={envelope.error} />}
        {envelope.data && (
          <>
            <JsonView value={envelope.data.payload} />
            <p className="mt-4 text-xs text-ink-soft" title={dateTime(envelope.data.created_at)}>
              Made from:{" "}
              {Object.entries(envelope.data.inputs)
                .map(([input, v]) => `${humanize(input)} v${v}`)
                .join(", ") || "nothing (a starting input)"}
            </p>
          </>
        )}
      </CardBody>
    </Card>
  );
}

/** Any stage, generically: its checkpoint, problems, outputs (every version) and activity. */
export function StagePage() {
  const { stageId = "" } = useParams();
  const { run } = useRunContext();
  const stage = run.stages.find((s) => s.id === stageId);
  if (!stage) return <Empty title="No such stage">This run has no stage called “{stageId}”.</Empty>;
  if (stage.agent === "literature") return <LiteraturePage run={run} stage={stage} />;
  if (stage.agent === "hypothesis") return <HypothesesPage run={run} stage={stage} />;
  const outputs = Object.entries(stage.output_versions);
  return (
    <div className="space-y-6">
      <header>
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-semibold">{stageTitle(stage.id)}</h1>
          <StatusBadge stage={stage} run={run} />
          {stage.attempt > 1 && <span className="text-sm text-ink-soft">attempt {stage.attempt}</span>}
          <div className="ml-auto">
            <ReopenButton
              run={run}
              stage={stage}
              label="Reopen the checkpoint"
              title={`Reopen the ${stageTitle(stage.id)} checkpoint?`}
              description="Approve it again as it is, or request changes to have this stage run again."
            />
          </div>
        </div>
        {STAGES[stage.id] && <p className="mt-2 max-w-3xl text-sm text-ink-soft">{STAGES[stage.id].description}</p>}
      </header>

      {stage.gate && <CheckpointCard run={run} gate={stage.gate} showStageLink={false} />}
      {stage.status === "failed" && <FailureCard run={run} stage={stage} />}
      {stage.status === "stale" && stage.stale_reason && (
        <p className="rounded-lg bg-stale-soft px-4 py-3 text-sm text-stale">Will run again: {stage.stale_reason}</p>
      )}

      {outputs.length ? (
        outputs.map(([name, version]) => <ArtifactPanel key={name} run={run} name={name} current={version} />)
      ) : (
        <Empty title="Nothing here yet">This stage has not produced anything yet.</Empty>
      )}

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_22rem]">
        <Card>
          <CardHeader title="Activity" />
          <CardBody>
            <ActivityFeed project={run.project} runId={run.run_id} stage={stage.id} limit={25} />
          </CardBody>
        </Card>
        <Card>
          <CardBody>
            <NoteBox run={run} stage={stage.id} />
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
