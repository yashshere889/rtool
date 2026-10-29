import { Plus } from "lucide-react";
import { Link } from "react-router";

import { runPath } from "@/api/client";
import { useRuns } from "@/api/hooks";
import type { RunSummary } from "@/api/types";
import { StageDots } from "@/components/StageStepper";
import { Badge } from "@/components/ui/badge";
import { buttonStyles } from "@/components/ui/button";
import { Empty, ErrorNote, Loading } from "@/components/ui/states";
import { dateTime, timeAgo } from "@/lib/format";
import { runHeadline } from "@/lib/stages";

function RunCard({ run }: { run: RunSummary }) {
  const headline = runHeadline(run);
  return (
    <Link
      to={runPath(run.project, run.run_id)}
      className="block rounded-xl border border-line bg-surface p-4 shadow-sm transition-colors hover:border-accent/50"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <p className="min-w-0 flex-1 font-semibold">{run.topic || run.run_id}</p>
        <Badge tone={headline.tone}>{headline.text}</Badge>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-ink-soft">
        <StageDots run={run} />
        <span title={dateTime(run.created_at)}>Started {timeAgo(run.created_at)}</span>
        <span>Run {run.run_id}</span>
        {run.error && <span className="text-bad">{run.error}</span>}
      </div>
    </Link>
  );
}

export function Projects() {
  const runs = useRuns();
  const groups = new Map<string, RunSummary[]>();
  for (const run of runs.data ?? []) groups.set(run.project, [...(groups.get(run.project) ?? []), run]);
  const needYou = (runs.data ?? []).filter((r) => r.open_gates.length > 0).length;

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Research projects</h1>
          <p className="mt-1 text-sm text-ink-soft">
            {needYou ? `${needYou} run${needYou === 1 ? " needs" : "s need"} your decision.` : "Every run, newest first."}
          </p>
        </div>
        <Link to="/new" className={buttonStyles({ size: "lg" })}>
          <Plus /> New research
        </Link>
      </div>

      {runs.isPending && <Loading label="Loading projects" />}
      {runs.isError && <ErrorNote error={runs.error} title="Could not load the projects" />}
      {runs.data && runs.data.length === 0 && (
        <Empty
          title="No research yet"
          action={
            <Link to="/new" className={buttonStyles()}>
              Start your first project
            </Link>
          }
        >
          Type a research question; the tool searches the literature, proposes hypotheses, plans and runs the
          experiment, and writes the report, asking you at every important step.
        </Empty>
      )}
      <div className="space-y-8">
        {[...groups.entries()].map(([project, items]) => (
          <section key={project} aria-labelledby={`project-${project}`}>
            <h2 id={`project-${project}`} className="mb-3 text-sm font-semibold tracking-wide text-ink-soft uppercase">
              {project}
            </h2>
            <div className="space-y-3">
              {items.map((run) => (
                <RunCard key={run.run_id} run={run} />
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
