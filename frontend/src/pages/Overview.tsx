import { CheckCircle2, CirclePause, Loader, Play, Square } from "lucide-react";

import { useResume, useSettings, useStop } from "@/api/hooks";
import type { RunDetail } from "@/api/types";
import { ActivityFeed } from "@/components/ActivityFeed";
import { CheckpointCard } from "@/components/CheckpointCard";
import { FailureCard } from "@/components/FailureCard";
import { useRunContext } from "@/components/layout/RunLayout";
import { QueuedActions } from "@/components/QueuedActions";
import { StageStepper } from "@/components/StageStepper";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { compactNumber, dateTime, timeAgo } from "@/lib/format";
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
  const finished = headline.tone === "ok";
  const working = run.worker.active || headline.tone === "info";
  const Icon = finished ? CheckCircle2 : working ? Loader : CirclePause;
  return (
    <Card>
      <CardBody className="flex flex-wrap items-center gap-4 py-5">
        <Icon className={`size-6 ${finished ? "text-ok" : working ? "animate-spin text-info [animation-duration:2.5s]" : "text-ink-soft"}`} />
        <div className="min-w-0 flex-1">
          <p className="font-semibold">{headline.text}</p>
          <p className="text-sm text-ink-soft">
            {finished
              ? "Every stage is done. The final report and the review are on the Review page."
              : run.worker.detail ||
                (working
                  ? "Nothing needs you right now; you will see a decision card here when it does."
                  : "Nothing is advancing this run at the moment.")}
          </p>
          {run.worker.last_error && <p className="mt-1 text-sm text-bad">{run.worker.last_error}</p>}
        </div>
        {!finished && !run.worker.active && (
          <Button variant="secondary" onClick={() => resume.mutate()} busy={resume.isPending}>
            <Play /> Resume
          </Button>
        )}
        {!finished && run.worker.active && settings.data && !settings.data.runs_locally && (
          <Button variant="secondary" onClick={() => stop.mutate(false)} busy={stop.isPending}>
            <Square /> Pause
          </Button>
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

export function Overview() {
  const { run } = useRunContext();
  const otherQueued = { ...run, queued_actions: run.queued_actions.filter((a) => !a.gate) };
  return (
    <div className="space-y-6">
      <header>
        <p className="text-sm text-ink-soft">
          Started <span title={dateTime(run.created_at)}>{timeAgo(run.created_at)}</span> · {run.pipeline} pipeline ·{" "}
          {run.pack === "ml_dl" ? "machine learning" : "any field"}
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
          <CardHeader title="Activity" description="What the agents did, newest first. Updates live." />
          <CardBody>
            <ActivityFeed project={run.project} runId={run.run_id} />
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
