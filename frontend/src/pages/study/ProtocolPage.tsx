import { CircleAlert, CircleCheck, FileText, Pencil, Undo2 } from "lucide-react";
import { useId, useState } from "react";

import { ApiError, fileUrl } from "@/api/client";
import { useDecide, useProtocol, useProtocolCheck, useUndo } from "@/api/hooks";
import type { Protocol, ProtocolView, RunDetail, StageSummary } from "@/api/types";
import { CheckpointCard } from "@/components/CheckpointCard";
import { FailureCard } from "@/components/FailureCard";
import { StatusBadge } from "@/components/StatusBadge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Field, NumberInput, Textarea } from "@/components/ui/field";
import { Empty, ErrorNote, Loading } from "@/components/ui/states";
import { timeAgo } from "@/lib/format";
import { STAGES } from "@/lib/stages";
import { useDebounced } from "@/lib/useDebounced";

import { PlanPage } from "../plan/PlanPage";
import { Section } from "../plan/PlanRead";

const ARTIFACT = "protocol";

function Checklist({ protocol }: { protocol: Protocol }) {
  return (
    <ul className="space-y-1.5 text-sm">
      {protocol.checklist.map((c) => (
        <li key={c.name} className={c.ok ? "flex gap-1.5 text-ok" : "flex gap-1.5 text-warn"}>
          {c.ok ? <CircleCheck className="mt-0.5 size-4 shrink-0" /> : <CircleAlert className="mt-0.5 size-4 shrink-0" />}
          <span>
            <span className="text-ink">{c.name}</span>
            {c.detail && <span className="block text-xs text-ink-soft">{c.detail}</span>}
          </span>
        </li>
      ))}
    </ul>
  );
}

function sampleText(p: Protocol): string {
  return p.paired
    ? `${p.n_per_condition} ${p.unit}s in all, each measured under every condition`
    : `${p.n_per_condition} ${p.unit}s per condition (${p.n_per_condition * p.conditions.length} in all)`;
}

/** The three things the researcher may change; everything that follows from them is worked out by the server. */
function EditDialog({ run, view, onClose }: { run: RunDetail; view: ProtocolView; onClose: () => void }) {
  const p = view.protocol;
  const [effect, setEffect] = useState<number | null>(p.effect_size);
  const [materials, setMaterials] = useState(p.materials.join("\n"));
  const [blinding, setBlinding] = useState(p.blinding);
  const [note, setNote] = useState("");
  const decide = useDecide(run.project, run.run_id);
  const ids = { effect: useId(), materials: useId(), blinding: useId(), note: useId() };
  const value = {
    ...p,
    effect_size: effect ?? 0,
    materials: materials.split("\n").map((m) => m.trim()).filter(Boolean),
    blinding,
  };
  const dirty = JSON.stringify([value.effect_size, value.materials, value.blinding]) !== JSON.stringify([p.effect_size, p.materials, p.blinding]);
  const settled = useDebounced(value, 500);
  const check = useProtocolCheck(run.project, run.run_id, view.head, settled, dirty);
  const checked = check.data;
  const ready = dirty && Boolean(checked?.ok) && !check.isFetching && JSON.stringify(settled) === JSON.stringify(value);

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent
        title="Edit the protocol"
        description="Change the effect size, the materials or the blinding. The sample size, the allocation table and the checklist are worked out again."
      >
        <div className="space-y-4">
          <Field label="Smallest effect worth detecting (Cohen's d)" htmlFor={ids.effect} hint="A smaller effect needs more units.">
            <NumberInput id={ids.effect} step="0.05" min={0} max={5} value={effect} onValue={setEffect} />
          </Field>
          <Field label="Materials (one per line)" htmlFor={ids.materials}>
            <Textarea id={ids.materials} rows={4} value={materials} onChange={(e) => setMaterials(e.target.value)} />
          </Field>
          <Field label="Blinding" htmlFor={ids.blinding} hint="Who does not know which condition a unit is in.">
            <Textarea id={ids.blinding} rows={2} className="min-h-0" value={blinding} onChange={(e) => setBlinding(e.target.value)} />
          </Field>
          <Field label="Why (optional, kept with the new version)" htmlFor={ids.note}>
            <Textarea id={ids.note} rows={2} className="min-h-0" value={note} onChange={(e) => setNote(e.target.value)} />
          </Field>
          <div aria-live="polite" className="rounded-lg bg-muted px-3 py-2 text-sm">
            {!dirty ? (
              <span className="text-ink-soft">Nothing changed yet.</span>
            ) : checked && !checked.ok ? (
              <span className="flex gap-1.5 text-bad">
                <CircleAlert className="mt-0.5 size-4 shrink-0" /> {checked.error}
              </span>
            ) : checked?.protocol ? (
              <div className="space-y-2">
                <p>
                  Sample size: <span className="font-medium">{sampleText(checked.protocol)}</span>
                  {checked.protocol.n_per_condition !== p.n_per_condition && ` (was ${p.n_per_condition})`}.
                </p>
                <Checklist protocol={checked.protocol} />
              </div>
            ) : (
              <span className="text-ink-soft">Checking…</span>
            )}
          </div>
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button
              disabled={!ready}
              busy={decide.isPending}
              onClick={() =>
                decide.mutate(
                  {
                    request: { kind: "edit", stage: view.stage, payload: { artifact: ARTIFACT, base_version: view.head, value, note: note.trim() } },
                    label: "Your protocol edits",
                  },
                  { onSuccess: onClose },
                )
              }
            >
              Apply my edits
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function ProtocolPage({ run, stage }: { run: RunDetail; stage: StageSummary }) {
  const protocol = useProtocol(run.project, run.run_id);
  const undo = useUndo(run.project, run.run_id);
  const [editing, setEditing] = useState(false);
  const [designing, setDesigning] = useState(false);
  const header = (
    <header>
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold">Study protocol</h1>
        <StatusBadge stage={stage} run={run} />
      </div>
      <p className="mt-2 max-w-3xl text-sm text-ink-soft">{STAGES.protocol.description}</p>
    </header>
  );
  if (protocol.isPending) return <div className="space-y-6">{header}<Loading label="Loading the protocol" /></div>;
  if (protocol.isError || !protocol.data) {
    const missing = protocol.error instanceof ApiError && protocol.error.status === 404;
    return (
      <div className="space-y-6">
        {header}
        {stage.status === "failed" && <FailureCard run={run} stage={stage} />}
        {missing ? (
          <Empty title="No protocol yet">It appears here once you have chosen a hypothesis and the protocol has been designed.</Empty>
        ) : (
          <ErrorNote error={protocol.error} />
        )}
      </div>
    );
  }
  const view = protocol.data;
  const { protocol: p, plan } = view;
  if (designing) {
    return (
      <div className="space-y-4">
        <Button variant="ghost" size="sm" onClick={() => setDesigning(false)}>
          ← Back to the protocol
        </Button>
        <PlanPage run={run} stage={stage} designOnly />
      </div>
    );
  }
  const queuedEdit = run.queued_actions.find(
    (a) => a.kind === "edit" && (a.payload.artifact === ARTIFACT || a.payload.artifact === "experiment_plan"),
  );
  return (
    <div className="space-y-6">
      {header}
      {stage.gate && <CheckpointCard run={run} gate={stage.gate} showStageLink={false} />}
      {stage.status === "failed" && <FailureCard run={run} stage={stage} />}
      {queuedEdit && (
        <div className="flex flex-wrap items-center gap-3 rounded-lg bg-info-soft px-4 py-3 text-sm text-info">
          Your edits to the protocol are on their way; they are applied when a worker picks them up.
          <Button variant="secondary" size="sm" busy={undo.isPending} onClick={() => undo.mutate(queuedEdit.id)}>
            <Undo2 /> Undo
          </Button>
        </div>
      )}
      <p className="text-sm text-ink-soft">
        Version {view.version}
        {view.by ? ` · edited by ${view.by.replace(/ \(web\)$/, "")} from version ${view.edited_from}` : " · written by the protocol agent"} ·{" "}
        {timeAgo(view.created_at)}
        {view.edit_note && <span className="block">“{view.edit_note}”</span>}
      </p>

      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="space-y-4">
          <Section title="The study">
            {view.hypothesis_title && <p className="text-sm text-ink-soft">Tests: {view.hypothesis_title}</p>}
            <p className="text-sm">{plan.objective}</p>
            <p className="text-sm text-ink-soft">{plan.design}</p>
            <h3 className="pt-1 text-sm font-semibold">Conditions (unit: {p.unit})</h3>
            <ul className="space-y-1 text-sm">
              {plan.conditions.map((c) => (
                <li key={c.name}>
                  <code className="font-mono">{c.name}</code> {c.is_baseline && <Badge tone="neutral">control</Badge>}
                  <span className="text-ink-soft"> · {c.description}</span>
                </li>
              ))}
            </ul>
            <h3 className="pt-1 text-sm font-semibold">Primary outcome (fixed before any data)</h3>
            <p className="text-sm">
              <code className="font-mono">{p.primary_metric}</code>
              <span className="text-ink-soft"> · compared with {plan.statistical_tests.map((t) => `${t.test}: ${t.comparison}`).join("; ")}</span>
            </p>
            <h3 className="pt-1 text-sm font-semibold">Procedure</h3>
            <ol className="list-decimal space-y-1 pl-5 text-sm">
              {plan.procedure.map((step) => (
                <li key={step}>{step}</li>
              ))}
            </ol>
            {plan.risks.length > 0 && (
              <>
                <h3 className="pt-1 text-sm font-semibold">Risks</h3>
                <ul className="list-disc space-y-1 pl-5 text-sm">
                  {plan.risks.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
              </>
            )}
          </Section>

          <Section title="Sample size and allocation">
            <p className="text-sm">
              <span className="font-medium">{sampleText(p)}</span> to detect d = {p.effect_size} with alpha {p.alpha} and power {p.power}.
            </p>
            <p className="text-sm text-ink-soft">
              Who gets which condition is a seeded random table of {view.allocation_rows} rows
              {p.allocation_path && (
                <>
                  {" "}
                  (<a className="text-accent underline underline-offset-2" href={fileUrl(run.project, run.run_id, p.allocation_path)} target="_blank" rel="noreferrer">whole table</a>)
                </>
              )}
              . The first rows:
            </p>
            {view.allocation_preview.length > 0 && (
              <div className="overflow-x-auto">
                <table className="w-full max-w-md text-sm">
                  <thead>
                    <tr className="text-left text-ink-soft">
                      {Object.keys(view.allocation_preview[0]).map((k) => (
                        <th key={k} className="py-1 pr-4 font-medium">{k}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {view.allocation_preview.map((row) => (
                      <tr key={row.unit} className="border-t border-line">
                        {Object.values(row).map((v, i) => (
                          <td key={i} className="py-1 pr-4 font-mono text-xs">{v}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Section>

          <Section title="Materials and blinding">
            <ul className="list-disc space-y-1 pl-5 text-sm">
              {p.materials.map((m) => (
                <li key={m}>{m}</li>
              ))}
            </ul>
            <p className="text-sm">
              <span className="font-medium">Blinding.</span> {p.blinding || <span className="text-warn">not stated</span>}
            </p>
          </Section>

          <Section title="The data to bring back">
            <p className="text-sm">{p.data_format}.</p>
          </Section>
        </div>

        <aside className="space-y-4 xl:sticky xl:top-20">
          <section className="space-y-2 rounded-xl border border-line bg-surface p-4">
            <h2 className="text-sm font-semibold">Checklist</h2>
            <Checklist protocol={p} />
          </section>
          {p.markdown_path && (
            <a href={fileUrl(run.project, run.run_id, p.markdown_path)} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-sm text-accent underline decoration-accent/40 underline-offset-2">
              <FileText className="size-4" /> Protocol as a document
            </a>
          )}
          {view.editable && !queuedEdit ? (
            <Button className="w-full" variant="secondary" onClick={() => setEditing(true)}>
              <Pencil /> Edit the protocol
            </Button>
          ) : (
            !queuedEdit && view.why_not_editable && <p className="text-sm text-ink-soft">Editing: {view.why_not_editable}.</p>
          )}
          {view.editable && !queuedEdit && (
            <Button className="w-full" variant="secondary" onClick={() => setDesigning(true)}>
              <Pencil /> Edit the design
            </Button>
          )}
          <p className="text-xs text-ink-soft">
            The design is the conditions, outcome, tests and procedure. Changing it works the sample size and allocation out again.
          </p>
          {view.by && <Badge tone="info">This version was edited by hand</Badge>}
        </aside>
      </div>
      {editing && <EditDialog run={run} view={view} onClose={() => setEditing(false)} />}
    </div>
  );
}
