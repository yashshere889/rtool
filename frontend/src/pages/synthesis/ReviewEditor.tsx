import { CircleAlert, CircleCheck, Plus, Trash2 } from "lucide-react";
import { useId, useState } from "react";

import { useDecide, useSynthesisCheck } from "@/api/hooks";
import type { RunDetail, SynthesisView } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Input, Textarea } from "@/components/ui/field";
import { useDebounced } from "@/lib/useDebounced";

import { Section } from "../plan/PlanRead";

type Sections = SynthesisView["synthesis"]["sections"];

/** Rewrite the review sentence by sentence. A sentence must cite findings the model was offered, and state only their numbers. */
export function ReviewEditor({ run, view, onClose }: { run: RunDetail; view: SynthesisView; onClose: () => void }) {
  const [sections, setSections] = useState<Sections>(() => structuredClone(view.synthesis.sections));
  const [note, setNote] = useState("");
  const decide = useDecide(run.project, run.run_id);
  const noteId = useId();
  const value = { ...view.synthesis, sections };
  const dirty = JSON.stringify(sections) !== JSON.stringify(view.synthesis.sections);
  const settled = useDebounced(value, 500);
  const check = useSynthesisCheck(run.project, run.run_id, view.head, settled, dirty);
  const ready = dirty && Boolean(check.data?.ok) && !check.isFetching && JSON.stringify(settled) === JSON.stringify(value);

  const change = (s: number, i: number, patch: Partial<Sections[number]["sentences"][number]>) =>
    setSections((all) =>
      all.map((section, si) =>
        si === s ? { ...section, sentences: section.sentences.map((x, xi) => (xi === i ? { ...x, ...patch } : x)) } : section,
      ),
    );

  return (
    <div className="space-y-4">
      <p className="rounded-lg bg-info-soft px-4 py-3 text-sm text-info">
        Each sentence cites findings by id (C1, C2…). It is kept only if every finding it cites was offered and verified in its
        paper, and every number it states is in them. The report is rebuilt from your sentences.
      </p>
      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="space-y-4">
          {sections.map((section, s) => (
            <Section key={section.title} title={section.title}>
              <ul className="space-y-4">
                {section.sentences.map((sentence, i) => (
                  <li key={i} className="space-y-2">
                    <Textarea
                      aria-label={`Sentence ${i + 1} of ${section.title}`}
                      rows={2}
                      className="min-h-0"
                      value={sentence.text}
                      onChange={(e) => change(s, i, { text: e.target.value })}
                    />
                    <div className="flex flex-wrap items-center gap-2">
                      <Input
                        aria-label={`Findings that sentence ${i + 1} of ${section.title} rests on`}
                        className="max-w-xs font-mono text-xs"
                        value={sentence.claims.join(", ")}
                        onChange={(e) => change(s, i, { claims: e.target.value.split(/[,\s]+/).filter(Boolean) })}
                      />
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() =>
                          setSections((all) => all.map((x, xi) => (xi === s ? { ...x, sentences: x.sentences.filter((_, k) => k !== i) } : x)))
                        }
                      >
                        <Trash2 /> Remove
                      </Button>
                    </div>
                  </li>
                ))}
              </ul>
              <Button
                variant="secondary"
                size="sm"
                onClick={() =>
                  setSections((all) =>
                    all.map((x, xi) => (xi === s ? { ...x, sentences: [...x.sentences, { text: "", claims: [], papers: [] }] } : x)),
                  )
                }
              >
                <Plus /> Add a sentence
              </Button>
            </Section>
          ))}
        </div>
        <aside className="space-y-4 xl:sticky xl:top-20">
          <section className="space-y-2 rounded-xl border border-line bg-surface p-4">
            <h2 className="text-sm font-semibold">Checks</h2>
            <div aria-live="polite" className="text-sm">
              {!dirty ? (
                <span className="text-ink-soft">Nothing changed yet.</span>
              ) : check.data && !check.data.ok ? (
                <span className="flex gap-1.5 text-bad">
                  <CircleAlert className="mt-0.5 size-4 shrink-0" /> {check.data.error}
                </span>
              ) : check.data?.ok ? (
                <span className="flex gap-1.5 text-ok">
                  <CircleCheck className="mt-0.5 size-4 shrink-0" /> Every sentence rests on verified findings.
                </span>
              ) : (
                <span className="text-ink-soft">Checking…</span>
              )}
            </div>
            <label htmlFor={noteId} className="block pt-1 text-xs font-medium text-ink-soft">
              Why (optional, kept with the new version)
            </label>
            <Textarea id={noteId} rows={2} className="min-h-0" value={note} onChange={(e) => setNote(e.target.value)} />
            <div className="flex flex-wrap gap-2 pt-1">
              <Button
                disabled={!ready}
                busy={decide.isPending}
                onClick={() =>
                  decide.mutate(
                    {
                      request: { kind: "edit", stage: view.stage, payload: { artifact: "synthesis", base_version: view.head, value, note: note.trim() } },
                      label: "Your review edits",
                    },
                    { onSuccess: onClose },
                  )
                }
              >
                Apply my edits
              </Button>
              <Button variant="ghost" onClick={onClose}>
                Discard
              </Button>
            </div>
          </section>
          <details className="rounded-xl border border-line bg-surface p-4 text-sm">
            <summary className="cursor-pointer font-semibold">Findings you can cite</summary>
            <ul className="mt-3 space-y-2">
              {Object.entries(view.synthesis.claims).map(([id, claim]) => (
                <li key={id}>
                  <span className="font-mono text-xs">{id}</span> · {claim.text}
                  <span className="block text-xs text-ink-soft italic">“{claim.quote}”</span>
                </li>
              ))}
            </ul>
          </details>
        </aside>
      </div>
    </div>
  );
}
