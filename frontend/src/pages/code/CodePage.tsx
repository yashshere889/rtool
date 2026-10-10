import { Check, CircleAlert, Loader2 } from "lucide-react";
import { useSearchParams } from "react-router";

import { ApiError } from "@/api/client";
import { useCode } from "@/api/hooks";
import type { CodeView, RunDetail, StageSummary } from "@/api/types";
import { CheckpointCard } from "@/components/CheckpointCard";
import { FailureCard } from "@/components/FailureCard";
import { StatusBadge } from "@/components/StatusBadge";
import { Empty, ErrorNote, Loading } from "@/components/ui/states";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/cn";
import { STEPS, stepStates, type StepState } from "@/lib/code";
import { STAGES } from "@/lib/stages";

import { CodeTab } from "./CodeTab";
import { ResultsTab } from "./ResultsTab";
import { RunsTab } from "./RunsTab";
import { TestsTab } from "./TestsTab";

const STEP_STYLE: Record<StepState, string> = {
  done: "border-ok/40 bg-ok-soft/50 text-ok",
  current: "border-info/40 bg-info-soft/60 text-info",
  attention: "border-warn/50 bg-warn-soft/60 text-warn",
  failed: "border-bad/50 bg-bad-soft/60 text-bad",
  waiting: "border-line text-ink-soft",
};

function StepStrip({ view }: { view: CodeView }) {
  const states = stepStates(view);
  return (
    <ol aria-label="Progress of this stage" className="grid gap-2 sm:grid-cols-5">
      {STEPS.map((step, i) => (
        <li
          key={step.id}
          aria-current={states[i] === "current" || states[i] === "attention" ? "step" : undefined}
          className={cn("flex items-center gap-2 rounded-lg border px-3 py-2 text-sm", STEP_STYLE[states[i]])}
        >
          {states[i] === "done" ? (
            <Check className="size-4 shrink-0" />
          ) : states[i] === "current" ? (
            <Loader2 className="size-4 shrink-0 animate-spin motion-reduce:animate-none" />
          ) : states[i] === "failed" || states[i] === "attention" ? (
            <CircleAlert className="size-4 shrink-0" />
          ) : (
            <span className="w-4 shrink-0 text-center text-xs">{i + 1}</span>
          )}
          <span className="leading-tight">{step.label}</span>
        </li>
      ))}
    </ol>
  );
}

function defaultTab(view: CodeView): string {
  if (view.status === "done") return "results";
  if (view.status === "suspended") return "runs";
  return "code";
}

export function CodePage({ run, stage }: { run: RunDetail; stage: StageSummary }) {
  const code = useCode(run.project, run.run_id);
  const [params, setParams] = useSearchParams();

  const header = (
    <header>
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold">Code & runs</h1>
        <StatusBadge stage={stage} run={run} />
      </div>
      <p className="mt-2 max-w-3xl text-sm text-ink-soft">{STAGES.code.description}</p>
    </header>
  );

  if (code.isPending) return <div className="space-y-6">{header}<Loading label="Loading the code" /></div>;
  if (code.isError) {
    const missing = code.error instanceof ApiError && code.error.status === 404;
    return (
      <div className="space-y-6">
        {header}
        {missing ? <Empty title="No coding stage">This run's pipeline has no coding stage.</Empty> : <ErrorNote error={code.error} />}
      </div>
    );
  }
  const view = code.data;
  const tab = params.get("tab") ?? defaultTab(view);
  const notStarted = view.status === "pending" && !view.phase;

  return (
    <div className="space-y-6">
      {header}
      {stage.gate && <CheckpointCard run={run} gate={stage.gate} showStageLink={false} />}
      {stage.status === "failed" && (
        <>
          <FailureCard run={run} stage={stage} />
          {view.editable && (
            <p className="text-sm text-ink-soft">
              You can fix experiment.py in the Code tab below and choose “Save and resume”: your version is tested first.
            </p>
          )}
        </>
      )}
      {stage.status === "stale" && stage.stale_reason && (
        <p className="rounded-lg bg-stale-soft px-4 py-3 text-sm text-stale">Will run again: {stage.stale_reason}</p>
      )}
      {notStarted ? (
        <Empty title="Not started yet">The coding agent starts once the plan is approved and the datasets are downloaded.</Empty>
      ) : (
        <>
          <StepStrip view={view} />
          <Tabs
            value={tab}
            onValueChange={(value) =>
              setParams((prev) => {
                const next = new URLSearchParams(prev);
                next.set("tab", value);
                return next;
              })
            }
          >
            <TabsList>
              <TabsTrigger value="code">Code</TabsTrigger>
              <TabsTrigger value="tests">Tests & repairs ({view.fixes.length})</TabsTrigger>
              <TabsTrigger value="runs">
                Runs ({view.runs.filter((r) => r.status === "ok").length}/{view.runs.length})
              </TabsTrigger>
              <TabsTrigger value="results">Results</TabsTrigger>
            </TabsList>
            <TabsContent value="code" className="mt-4">
              <CodeTab run={run} view={view} />
            </TabsContent>
            <TabsContent value="tests" className="mt-4">
              <TestsTab run={run} view={view} />
            </TabsContent>
            <TabsContent value="runs" className="mt-4">
              <RunsTab run={run} view={view} />
            </TabsContent>
            <TabsContent value="results" className="mt-4">
              <ResultsTab run={run} />
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  );
}
