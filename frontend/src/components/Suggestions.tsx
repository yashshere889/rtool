import { Plus } from "lucide-react";

import { useEvents } from "@/api/hooks";
import type { RunDetail } from "@/api/types";
import { stageSuggestions } from "@/lib/events";

/** One-click requests for a "request changes" comment: from what the stage reported, plus ``extra`` from the page.
 * A click adds the request to the comment on a line of its own; one already in the comment is greyed out. */
export function Suggestions({
  run,
  stage,
  extra = [],
  value,
  onChange,
}: {
  run: RunDetail;
  stage: string;
  extra?: string[];
  value: string;
  onChange: (value: string) => void;
}) {
  const events = useEvents(run.project, run.run_id);
  const items = [...new Set([...stageSuggestions(events.data?.events ?? [], stage), ...extra])];
  if (!items.length) return null;
  return (
    <div className="mt-3">
      <p className="text-xs font-medium text-ink-soft">Suggestions from what the tool found</p>
      <ul className="mt-1.5 flex flex-wrap gap-1.5">
        {items.map((item) => {
          const added = value.includes(item);
          return (
            <li key={item}>
              <button
                type="button"
                disabled={added}
                onClick={() => onChange(value.trim() ? `${value.trim()}\n${item}` : item)}
                className="inline-flex items-start gap-1 rounded-md border border-line bg-surface px-2 py-1 text-left text-xs hover:border-accent/50 disabled:opacity-50"
              >
                <Plus className="mt-0.5 size-3 shrink-0" aria-hidden /> {item}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
