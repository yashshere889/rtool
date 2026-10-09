import { useId, useMemo, useState } from "react";

import type { HypothesisCardData, OpenGate, RunDetail } from "@/api/types";
import { useGateDecision } from "@/components/CheckpointCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent } from "@/components/ui/dialog";
import { Field, Input, Textarea } from "@/components/ui/field";
import { COMPUTE, EDITABLE, changedFields, draftProblems, fieldLabel, toDraft, type Draft } from "@/lib/hypotheses";

/** Choose a hypothesis at the checkpoint, as it is or after editing it, with an optional note for the planner. */
export function ChooseDialog({
  run,
  gate,
  card,
  editing,
  open,
  onOpenChange,
}: {
  run: RunDetail;
  gate: OpenGate;
  card: HypothesisCardData;
  editing: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const h = card.hypothesis;
  const index = card.index ?? 0;
  const { send, decide } = useGateDecision(run, gate);
  const [draft, setDraft] = useState<Draft>(() => toDraft(h));
  const [note, setNote] = useState("");
  const noteId = useId();
  const formId = useId();
  const edit = useMemo(() => (editing ? changedFields(h, draft) : {}), [editing, h, draft]);
  const problems = editing ? draftProblems(draft) : [];
  const changed = Object.keys(edit);

  function submit() {
    const payload: Record<string, unknown> = { index, note: note.trim() };
    if (changed.length) payload.edit = edit;
    const label = `Choose option ${index + 1}${changed.length ? " (edited)" : ""}`;
    send("select", payload, label, { onSuccess: () => onOpenChange(false) });
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        title={editing ? `Edit and choose option ${index + 1}` : `Choose option ${index + 1}?`}
        description={
          editing
            ? "Change what you need; the experiment is planned for your version. Evidence and scores stay as the agent found them."
            : "The experiment will be planned for this hypothesis. You can undo while it waits for a worker, and reopen the choice later."
        }
        className={editing ? "max-h-[90vh] w-[min(48rem,calc(100vw-2rem))] overflow-y-auto" : undefined}
      >
        <form
          id={formId}
          className="space-y-4"
          onSubmit={(event) => {
            event.preventDefault();
            if (!problems.length) submit();
          }}
        >
          {editing ? (
            EDITABLE.map((field) => {
              const id = `${formId}-${field.key}`;
              const label = (
                <span className="inline-flex items-center gap-2">
                  {field.label}
                  {field.key in edit && <Badge tone="info">changed</Badge>}
                </span>
              );
              const value = draft[field.key] ?? "";
              const set = (next: string) => setDraft((prev) => ({ ...prev, [field.key]: next }));
              return (
                <Field key={field.key} label={label} hint={field.hint} htmlFor={id}>
                  {field.kind === "line" ? (
                    <Input id={id} value={value} onChange={(e) => set(e.target.value)} />
                  ) : field.kind === "compute" ? (
                    <select
                      id={id}
                      className="h-9 w-full rounded-lg border border-line bg-surface px-3 text-sm"
                      value={value}
                      onChange={(e) => set(e.target.value)}
                    >
                      {Object.entries(COMPUTE).map(([key, c]) => (
                        <option key={key} value={key}>
                          {c.label} ({c.detail})
                        </option>
                      ))}
                    </select>
                  ) : (
                    <Textarea
                      id={id}
                      rows={field.kind === "list" ? Math.max(2, value.split("\n").length) : 3}
                      className="min-h-0"
                      value={value}
                      onChange={(e) => set(e.target.value)}
                    />
                  )}
                </Field>
              );
            })
          ) : (
            <div className="rounded-lg bg-muted px-3 py-2 text-sm">
              <p className="font-medium">{h.title}</p>
              <p className="mt-1 leading-relaxed">{h.statement}</p>
            </div>
          )}
          <Field
            label="A note for the planner (optional)"
            hint="The planner reads it when it designs the experiment, e.g. “prefer public datasets” or “keep it under 4 GPU-hours”."
            htmlFor={noteId}
          >
            <Textarea id={noteId} rows={2} className="min-h-0" value={note} onChange={(e) => setNote(e.target.value)} />
          </Field>
          {problems.length > 0 && (
            <ul className="list-disc rounded-lg bg-bad-soft py-2 pr-3 pl-8 text-sm text-bad">
              {problems.map((problem) => (
                <li key={problem}>{problem}</li>
              ))}
            </ul>
          )}
          {editing && (
            <p className="text-sm text-ink-soft">
              {changed.length ? `You changed: ${changed.map(fieldLabel).join(", ")}.` : "Nothing changed yet."}
            </p>
          )}
          <div className="flex justify-end gap-2">
            <DialogClose asChild>
              <Button variant="ghost">Cancel</Button>
            </DialogClose>
            <Button type="submit" busy={decide.isPending} disabled={problems.length > 0}>
              {editing && changed.length ? "Choose my version" : "Choose this hypothesis"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
