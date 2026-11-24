import { CircleAlert, CircleCheck, CircleMinus } from "lucide-react";
import { useState } from "react";

import { useAgents, useDoctor, useFields, usePipelines } from "@/api/hooks";
import type { CheckInfo, FieldInfo } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { ErrorNote, Loading } from "@/components/ui/states";
import { humanize } from "@/lib/format";
import type { Tone } from "@/lib/stages";

const FIELD_STATE: Record<FieldInfo["state"], { label: string; tone: Tone }> = {
  offered: { label: "Offered", tone: "ok" },
  not_offered: { label: "Not offered here", tone: "neutral" },
  shadowed: { label: "Never used", tone: "warn" },
  broken: { label: "Broken", tone: "bad" },
};

const PIPELINE_WORDS: Record<string, string> = {
  research: "an experiment on the cluster",
  study: "a study you carry out",
  synthesis: "a literature review only",
};

/** The fields (domain packs): what each is for, the pipeline its runs use, and whether this server offers it. */
export function FieldsTab() {
  const fields = useFields();
  if (fields.isPending) return <Loading />;
  if (fields.isError) return <ErrorNote error={fields.error} />;
  return (
    <Card>
      <CardHeader title="Fields" description="A field gives the agents their starting point, and decides which pipeline a new run uses." />
      <CardBody>
        <ul className="divide-y divide-line">
          {fields.data.map((f) => {
            const state = FIELD_STATE[f.state];
            return (
              <li key={f.name} className="space-y-1 py-3 text-sm">
                <p className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{f.title || f.name}</span>
                  <code className="font-mono text-xs text-ink-soft">{f.name}</code>
                  <Badge tone={state.tone}>{state.label}</Badge>
                  {f.pipeline && <Badge>{f.pipeline} pipeline</Badge>}
                  {f.max_gpu_hours !== null && <Badge tone="info">at most {f.max_gpu_hours} GPU-hours</Badge>}
                </p>
                {f.description && <p className="text-ink-soft">{f.description}</p>}
                {f.detail && <p className="text-warn">{f.detail}</p>}
                <p className="text-xs text-ink-soft">From {f.origin}</p>
              </li>
            );
          })}
        </ul>
      </CardBody>
    </Card>
  );
}

/** Each pipeline as its ordered stages, and the agents those stages use. */
export function PipelinesTab() {
  const pipelines = usePipelines();
  const agents = useAgents();
  if (pipelines.isPending || agents.isPending) return <Loading />;
  if (pipelines.isError || agents.isError) return <ErrorNote error={pipelines.error ?? agents.error} />;
  return (
    <div className="space-y-6">
      {pipelines.data.map((p) => (
        <Card key={p.name}>
          <CardHeader title={`${humanize(p.name)} pipeline`} description={`${PIPELINE_WORDS[p.name] ?? ""}${PIPELINE_WORDS[p.name] ? ". " : ""}${p.description}`} />
          <CardBody>
            <ol className="space-y-1.5 text-sm">
              {p.stages.map((s, i) => (
                <li key={s.id} className="flex flex-wrap items-baseline gap-x-2">
                  <span className="w-5 text-right text-xs text-ink-soft tabular-nums">{i + 1}</span>
                  <span className="font-medium">{humanize(s.id)}</span>
                  <code className="font-mono text-xs text-ink-soft">{s.agent}</code>
                  {s.gate && <Badge tone="info">checkpoint: {humanize(s.gate)}</Badge>}
                </li>
              ))}
            </ol>
          </CardBody>
        </Card>
      ))}
      <Card>
        <CardHeader title="Agents" description="What each stage's agent reads and writes." />
        <CardBody>
          <ul className="divide-y divide-line">
            {agents.data.map((a) => (
              <li key={a.name} className="space-y-0.5 py-2.5 text-sm">
                <p className="flex flex-wrap items-center gap-2">
                  <code className="font-mono font-medium">{a.name}</code>
                  <Badge tone={a.uses_llm ? "info" : "neutral"}>{a.uses_llm ? "uses the model" : "no model"}</Badge>
                </p>
                <p className="text-ink-soft">{a.description}</p>
                <p className="text-xs text-ink-soft">
                  Reads {[...a.consumes, ...a.optional.map((o) => `${o} (if any)`)].join(", ") || "nothing"} · writes {a.produces.join(", ") || "nothing"}
                </p>
              </li>
            ))}
          </ul>
        </CardBody>
      </Card>
    </div>
  );
}

const CHECK_ICON: Record<CheckInfo["status"], { icon: typeof CircleCheck; tone: string }> = {
  ok: { icon: CircleCheck, tone: "text-ok" },
  warn: { icon: CircleAlert, tone: "text-warn" },
  fail: { icon: CircleAlert, tone: "text-bad" },
  skip: { icon: CircleMinus, tone: "text-ink-soft" },
};

/** `rtool doctor`: whether this machine can run the pipeline. The paper APIs are only contacted when asked. */
export function ChecksTab() {
  const [network, setNetwork] = useState<boolean | null>(null); // null: nothing asked for yet
  const doctor = useDoctor(network ?? false, network !== null);
  return (
    <Card>
      <CardHeader
        title="Checks of this machine"
        description="Whether the server can run the pipeline with its configuration: the scheduler, the model, disk quotas, and (when asked) the paper APIs."
        action={
          <div className="flex gap-2">
            <Button variant="secondary" size="sm" busy={doctor.isFetching && network === false} onClick={() => setNetwork(false)}>
              Run the checks
            </Button>
            <Button variant="secondary" size="sm" busy={doctor.isFetching && network === true} onClick={() => setNetwork(true)}>
              Also reach the paper APIs
            </Button>
          </div>
        }
      />
      <CardBody>
        {network === null ? (
          <p className="text-sm text-ink-soft">Nothing has been checked yet. The checks ask the scheduler and the disks, so they run when you ask.</p>
        ) : doctor.isPending ? (
          <Loading label="Checking" />
        ) : doctor.isError ? (
          <ErrorNote error={doctor.error} />
        ) : (
          <ul className="space-y-2 text-sm">
            {doctor.data.map((c) => {
              const { icon: Icon, tone } = CHECK_ICON[c.status];
              return (
                <li key={c.name} className="flex gap-2">
                  <Icon className={`mt-0.5 size-4 shrink-0 ${tone}`} aria-label={c.status} />
                  <span>
                    <span className="font-medium">{c.name}</span> <span className="text-ink-soft">{c.detail}</span>
                  </span>
                </li>
              );
            })}
          </ul>
        )}
      </CardBody>
    </Card>
  );
}
