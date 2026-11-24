import { CirclePause, Loader, Play, Square } from "lucide-react";
import { useState } from "react";

import { useResume, useRunStats, useSettings, useStop } from "@/api/hooks";
import type { RunDetail } from "@/api/types";
import { ActivityFeed } from "@/components/ActivityFeed";
import { CheckpointCard } from "@/components/CheckpointCard";
import { FailureCard } from "@/components/FailureCard";
import { FinishedSummary } from "@/components/FinishedSummary";
import { useRunContext } from "@/components/layout/RunLayout";
import { QueuedActions } from "@/components/QueuedActions";
import { RunTools } from "@/components/RunTools";
import { StageProgress } from "@/components/StageProgress";
import { StageStepper } from "@/components/StageStepper";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { compactNumber, dateTime, duration, timeAgo } from "@/lib/format";
import { runHeadline, stageTitle } from "@/lib/stages";

function RightNow({ run }: { run: RunDetail }) {
  const settings = useSettings();
  const resume = useResume(run.project, run.run_id);
  const stop = useStop(run.project, run.run_id);
  const headline = runHeadline(run);
  const failed = run.stages.filter((s) => s.status === "failed");

  if (run.open_gates.length) {
    return (
      <div className="space-y-4">
        {run.open_gates.map((gate) => (
          <CheckpointCard key={`${gate.stage}-${gate.nonce}`} run={run} gate={gate} />
        ))}
      </div>
    );
  }
  if (failed.length) {
    return (
      <div className="space-y-4">
        {failed.map((stage) => (
          <FailureCard key={stage.id} run={run} stage={stage} />
        ))}
      </div>
    );
  }
  if (headline.tone === "ok") return <FinishedSummary run={run} />;
  const working = run.worker.active || headline.tone === "info";
  const running = run.stages.find((s) => s.id === headline.stage);
  const Icon = working ? Loader : CirclePause;
  return (
    <Card>
      <CardBody className="flex flex-wrap items-center gap-4 py-5">
        <Icon className={`size-6 ${working ? "animate-spin text-info [animation-duration:2.5s]" : "text-ink-soft"}`} />
        <div className="min-w-0 flex-1">
          <p className="font-semibold">{headline.text}</p>
          <p className="text-sm text-ink-soft">
            {run.worker.detail ||
              (working
                ? "Nothing needs you right now; you will see a decision card here when it does."
                : "Nothing is advancing this run at the moment.")}
          </p>
          {run.worker.last_error && <p className="mt-1 text-sm text-bad">{run.worker.last_error}</p>}
        </div>
        {!run.worker.active && (
          <Button variant="secondary" onClick={() => resume.mutate()} busy={resume.isPending}>
            <Play /> Resume
          </Button>
        )}
        {run.worker.active && settings.data && !settings.data.runs_locally && (
          <Button variant="secondary" onClick={() => stop.mutate(false)} busy={stop.isPending}>
            <Square /> Pause
          </Button>
        )}
        {working && running && (
          <div className="w-full border-t border-line pt-4">
            <StageProgress run={run} stage={running} />
          </div>
        )}
      </CardBody>
    </Card>
  );
}

function Usage({ run }: { run: RunDetail }) {
  const usage = (run.run_state?.usage ?? {}) as Record<string, number>;
  const items = [
    ["AI model calls", compactNumber(usage.llm_calls ?? 0)],
    ["Tokens read / written", `${compactNumber(usage.prompt_tokens ?? 0)} / ${compactNumber(usage.completion_tokens ?? 0)}`],
    ["GPU-hours requested", (usage.gpu_hours ?? 0).toLocaleString(undefined, { maximumFractionDigits: 1 })],
    ["Cluster jobs", compactNumber(usage.jobs ?? 0)],
    ["Worker sessions", String(run.run_state?.episodes ?? 0)],
  ];
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
      {items.map(([label, value]) => (
        <div key={label}>
          <dt className="text-ink-soft">{label}</dt>
          <dd className="font-semibold">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Where the run's wall time went, split into who was working: the tool, a person, the cluster queue. */
function TimeSpent({ run }: { run: RunDetail }) {
  const stats = useRunStats(run.project, run.run_id);
  if (!stats.data) return <p className="text-sm text-ink-soft">{stats.isError ? "Nothing recorded yet." : "Loading…"}</p>;
  const w = stats.data.wall;
  const parts: [string, number][] = [
    ["The tool was working", w.running_s],
    ["Waiting for a decision", w.person_s],
    ["Waiting for cluster jobs", w.jobs_s],
    ["In the cluster queue", w.queue_s],
    ["A failed stage waited for you", w.failed_s],
  ];
  return (
    <div className="space-y-3 text-sm">
      <p>
        <span className="text-ink-soft">Since the start:</span> <span className="font-semibold">{duration(w.total_s)}</span>
      </p>
      <dl className="space-y-1">
        {parts
          .filter(([, seconds]) => seconds >= 1)
          .map(([label, seconds]) => (
            <div key={label} className="flex justify-between gap-3">
              <dt className="text-ink-soft">{label}</dt>
              <dd className="tabular-nums">{duration(seconds)}</dd>
            </div>
          ))}
      </dl>
      <details>
        <summary className="cursor-pointer text-ink-soft">By stage</summary>
        <dl className="mt-2 space-y-1">
          {stats.data.stages.map((s) => (
            <div key={s.stage} className="flex justify-between gap-3">
              <dt>{stageTitle(s.stage)}</dt>
              <dd className="tabular-nums">{duration(s.active_s)}</dd>
            </div>
          ))}
        </dl>
      </details>
    </div>
  );
}

export function Overview() {
  const { run } = useRunContext();
  const [feedStage, setFeedStage] = useState("");
  const otherQueued = { ...run, queued_actions: run.queued_actions.filter((a) => !a.gate) };
  return (
    <div className="space-y-6">
      <header>
        <p className="text-sm text-ink-soft">
          Started <span title={dateTime(run.created_at)}>{timeAgo(run.created_at)}</span> · {run.pipeline} pipeline ·{" "}
          {run.pack === "generic" ? "any field" : `${run.pack} pack`}
        </p>
        <h1 className="mt-1 text-2xl leading-snug font-semibold">{run.topic || run.run_id}</h1>
      </header>

      <section aria-label="Right now" className="space-y-3">
        <RightNow run={run} />
        <QueuedActions run={otherQueued} />
      </section>

      <section aria-labelledby="stages-heading">
        <h2 id="stages-heading" className="mb-3 text-sm font-semibold tracking-wide text-ink-soft uppercase">
          Stages
        </h2>
        <StageStepper run={run} />
      </section>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_20rem]">
        <Card>
          <CardHeader
            title="Activity"
            description="What the agents did, newest first. Updates live."
            action={
              <select
                aria-label="Show activity of"
                className="h-8 rounded-md border border-line bg-surface px-2 text-sm"
                value={feedStage}
                onChange={(e) => setFeedStage(e.target.value)}
              >
                <option value="">Every stage</option>
                {run.stages.map((s) => (
                  <option key={s.id} value={s.id}>
                    {stageTitle(s.id)}
                  </option>
                ))}
              </select>
            }
          />
          <CardBody>
            <ActivityFeed project={run.project} runId={run.run_id} stage={feedStage || undefined} />
          </CardBody>
        </Card>
        <div className="space-y-6">
          <Card>
            <CardHeader title="Usage" />
            <CardBody>
              <Usage run={run} />
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="Where the time went" />
            <CardBody>
              <TimeSpent run={run} />
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="This run as a whole" />
            <CardBody>
              <RunTools run={run} />
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="About this run" />
            <CardBody className="space-y-1 text-sm">
              <p>
                <span className="text-ink-soft">Project:</span> {run.project}
              </p>
              <p>
                <span className="text-ink-soft">Run:</span> {run.run_id}
              </p>
              <p>
                <span className="text-ink-soft">Current stage:</span>{" "}
                {stageTitle(run.stages.find((s) => s.status !== "done")?.id ?? "") || "none (finished)"}
              </p>
              <p>
                <span className="text-ink-soft">Tool version:</span> {run.tool_version}
              </p>
            </CardBody>
          </Card>
        </div>
      </div>
    </div>
  );
}
