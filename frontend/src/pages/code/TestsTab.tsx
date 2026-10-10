import { useState } from "react";

import type { CodeView, RunDetail } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { dateTime, fileSize, timeAgo } from "@/lib/format";
import { num, repairLabel } from "@/lib/code";

import { LogDialog } from "./LogDialog";
import { Section } from "../plan/PlanRead";

const PHASES: Record<string, string> = { check: "before running", smoke: "CPU test", full: "cluster runs" };

/** The CPU smoke tests, what the tool repaired along the way, and the logs. */
export function TestsTab({ run, view }: { run: RunDetail; view: CodeView }) {
  const [log, setLog] = useState<{ name: string; title: string } | null>(null);
  const smokeLog = (condition: string) => view.logs.find((l) => l.name === `logs/smoke-${condition}.log`);
  return (
    <div className="space-y-4">
      <Section title="Smoke tests on a CPU">
        <p className="text-sm text-ink-soft">
          Each condition runs once on a tiny problem before any GPU time is asked for, to catch crashes and wrong
          outputs early.
        </p>
        {view.smoke.length === 0 ? (
          <p className="text-sm text-ink-soft">No smoke test has run yet in this attempt.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[32rem] text-sm">
              <thead>
                <tr className="border-b border-line text-left text-xs text-ink-soft">
                  <th scope="col" className="py-2 pr-3 font-medium">Condition</th>
                  <th scope="col" className="py-2 pr-3 font-medium">Result</th>
                  <th scope="col" className="py-2 pr-3 font-medium">Time</th>
                  <th scope="col" className="py-2 pr-3 font-medium">Metrics (tiny problem)</th>
                  <th scope="col" className="py-2 font-medium">Log</th>
                </tr>
              </thead>
              <tbody>
                {view.smoke.map((s) => {
                  const file = smokeLog(s.condition);
                  return (
                    <tr key={s.condition} className="border-b border-line align-top last:border-0">
                      <td className="py-2 pr-3 font-mono text-xs">{s.condition}</td>
                      <td className="py-2 pr-3">
                        <Badge tone={s.ok ? "ok" : "bad"}>{s.ok ? "Passed" : "Failed"}</Badge>
                        {s.error && <span className="mt-1 block text-xs text-bad">{s.error}</span>}
                      </td>
                      <td className="py-2 pr-3 tabular-nums">{s.seconds.toFixed(1)} s</td>
                      <td className="py-2 pr-3 text-xs">
                        {Object.entries(s.metrics).map(([k, v]) => `${k} = ${num(v)}`).join(", ") || "–"}
                      </td>
                      <td className="py-2">
                        {file && (
                          <button type="button" className="text-sm text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent" onClick={() => setLog({ name: file.name, title: file.label })}>
                            Open
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      <Section title="What the tool repaired">
        {view.fixes.length === 0 ? (
          <p className="text-sm text-ink-soft">Nothing needed repairing in this attempt.</p>
        ) : (
          <ol className="space-y-2">
            {view.fixes.map((f, i) => (
              <li key={i} className="rounded-lg border border-line bg-canvas/40 px-3 py-2 text-sm">
                <p className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{repairLabel(f.signature)}</span>
                  <Badge>{PHASES[f.phase] ?? f.phase}</Badge>
                  <span className="text-xs text-ink-soft" title={dateTime(f.ts)}>
                    {timeAgo(f.ts)}
                  </span>
                </p>
                <p className="mt-0.5">{f.action}</p>
                {f.detail && <p className="mt-0.5 font-mono text-xs break-words text-ink-soft">{f.detail}</p>}
              </li>
            ))}
          </ol>
        )}
      </Section>

      <Section title="Logs">
        {view.logs.length === 0 ? (
          <p className="text-sm text-ink-soft">No logs yet.</p>
        ) : (
          <ul className="space-y-1 text-sm">
            {view.logs.map((l) => (
              <li key={l.name}>
                <button type="button" className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent" onClick={() => setLog({ name: l.name, title: l.label })}>
                  {l.label}
                </button>{" "}
                <span className="text-xs text-ink-soft">({fileSize(l.size)})</span>
              </li>
            ))}
          </ul>
        )}
      </Section>
      <LogDialog run={run} name={log?.name ?? null} title={log?.title ?? ""} onClose={() => setLog(null)} />
    </div>
  );
}
