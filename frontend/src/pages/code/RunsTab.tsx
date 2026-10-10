import { CircleStop } from "lucide-react";
import { useState } from "react";

import { useStop } from "@/api/hooks";
import type { CodeView, RunCell, RunDetail } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent } from "@/components/ui/dialog";
import { cn } from "@/lib/cn";
import { num, RUN_STATES } from "@/lib/code";
import { dateTime, timeAgo } from "@/lib/format";
import { TONE_STYLES } from "@/lib/stages";

import { LogDialog } from "./LogDialog";
import { Section } from "../plan/PlanRead";

function cellTitle(cell: RunCell): string {
  const parts = [`${cell.condition}, seed ${cell.seed}: ${RUN_STATES[cell.status].label}`];
  if (Object.keys(cell.metrics).length) parts.push(Object.entries(cell.metrics).map(([k, v]) => `${k} = ${num(v)}`).join(", "));
  if (cell.error) parts.push(cell.error);
  return parts.join("\n");
}

/** Every planned run (condition × seed), the cluster jobs, and a way to stop them. */
export function RunsTab({ run, view }: { run: RunDetail; view: CodeView }) {
  const [log, setLog] = useState<RunCell | null>(null);
  const [stopping, setStopping] = useState(false);
  const stop = useStop(run.project, run.run_id);
  const done = view.runs.filter((r) => r.status === "ok").length;
  const first = view.metrics[0];
  const byKey = new Map(view.runs.map((r) => [`${r.condition}|${r.seed}`, r]));
  return (
    <div className="space-y-4">
      <Section title="Runs">
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <span>
            {done} of {view.runs.length} runs finished
            {first ? <span className="text-ink-soft"> · cells show {first}</span> : null}
          </span>
          {view.status === "suspended" && (
            <Button variant="danger" size="sm" className="ml-auto" onClick={() => setStopping(true)}>
              <CircleStop /> Cancel the running jobs
            </Button>
          )}
        </div>
        {view.runs.length === 0 ? (
          <p className="text-sm text-ink-soft">No runs are planned yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="text-sm">
              <thead>
                <tr className="text-left text-xs text-ink-soft">
                  <th scope="col" className="py-2 pr-4 font-medium">Condition</th>
                  {view.seeds.map((seed) => (
                    <th key={seed} scope="col" className="px-1 py-2 text-center font-medium">
                      seed {seed}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {view.conditions.map((condition) => (
                  <tr key={condition}>
                    <th scope="row" className="py-1 pr-4 text-left font-mono text-xs font-normal">
                      {condition} {condition === view.baseline && <Badge tone="info">baseline</Badge>}
                    </th>
                    {view.seeds.map((seed) => {
                      const cell = byKey.get(`${condition}|${seed}`);
                      if (!cell) return <td key={seed} />;
                      const state = RUN_STATES[cell.status];
                      return (
                        <td key={seed} className="p-1">
                          <button
                            type="button"
                            title={cellTitle(cell)}
                            aria-label={cellTitle(cell)}
                            disabled={!cell.log}
                            onClick={() => setLog(cell)}
                            className={cn("block min-w-20 rounded-md px-2 py-1.5 text-center text-xs tabular-nums enabled:hover:ring-2 enabled:hover:ring-accent/30", TONE_STYLES[state.tone])}
                          >
                            {cell.status === "ok" && first ? num(cell.metrics[first]) : state.label}
                          </button>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="text-xs text-ink-soft">Choose a run to read its cluster log.</p>
      </Section>

      <Section title="Cluster jobs">
        {view.jobs.length === 0 ? (
          <p className="text-sm text-ink-soft">Nothing has been submitted yet: the runs start after you approve GPU time.</p>
        ) : (
          <ul className="space-y-1 text-sm">
            {view.jobs.map((job) => (
              <li key={job.id} className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-xs">job {job.id}</span>
                <span>
                  {job.runs} runs on {job.partition}, up to {job.gpu_hours} GPU-hours
                </span>
                <Badge tone={job.state === "COMPLETED" ? "ok" : job.state ? "warn" : "info"}>{job.state ? job.state.toLowerCase() : "queued or running"}</Badge>
                <span className="text-xs text-ink-soft" title={dateTime(job.submitted_at)}>
                  submitted {timeAgo(job.submitted_at)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Section>

      <LogDialog
        run={run}
        name={log?.log ?? null}
        title={log ? `Log of ${log.condition}, seed ${log.seed}` : ""}
        onClose={() => setLog(null)}
      />
      <Dialog open={stopping} onOpenChange={setStopping}>
        <DialogContent title="Cancel the running jobs?" description="Finished runs are kept and analysed; the others are reported as missing.">
          <div className="flex justify-end gap-2">
            <DialogClose asChild>
              <Button variant="ghost">Keep them running</Button>
            </DialogClose>
            <Button
              variant="danger"
              busy={stop.isPending}
              onClick={() => stop.mutate(true, { onSuccess: () => setStopping(false) })}
            >
              Cancel the jobs
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
