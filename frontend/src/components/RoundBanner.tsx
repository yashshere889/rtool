import { History } from "lucide-react";
import { useState, type ReactNode } from "react";

import type { Round } from "@/api/types";
import { humanize, timeAgo } from "@/lib/format";
import { stageTitle } from "@/lib/stages";

const SHOWN = 3;
const quote = (titles: string[]) => titles.map((t) => `“${t}”`).join(", ");

/** The first few titles, the rest behind "and N more" (a real keep/drop round can name dozens). */
export function TitleList({ titles }: { titles: string[] }) {
  const [all, setAll] = useState(false);
  if (all || titles.length <= SHOWN + 1) return <>{quote(titles)}</>;
  return (
    <>
      {quote(titles.slice(0, SHOWN))}{" "}
      <button type="button" className="text-info underline underline-offset-2 hover:no-underline" onClick={() => setAll(true)}>
        and {titles.length - SHOWN} more
      </button>
    </>
  );
}

/** Why this version replaced the previous one: the comments it acted on (or that earlier results changed).
 * ``unchanged``: the stage acted on the comments and produced the same version, so there is nothing to compare. */
export function RoundBanner({ round, what, children }: { round: Round; what: string; children?: ReactNode }) {
  const lines = round.feedback.flatMap((f, i) => {
    const who = f.by === "you" ? "You" : stageTitle(f.from_stage ?? "") || "A later stage";
    const out: ReactNode[] = [];
    if (f.text) out.push(<li key={`${i}-t`}>{who} asked: “{f.text}”</li>);
    if (f.dropped.length)
      out.push(
        <li key={`${i}-d`}>
          {who} dropped {f.dropped.length} {f.dropped.length === 1 ? "paper" : "papers"}: <TitleList titles={f.dropped} />
        </li>,
      );
    if (f.kept.length)
      out.push(
        <li key={`${i}-k`}>
          {who} kept {f.kept.length} {f.kept.length === 1 ? "paper" : "papers"}: <TitleList titles={f.kept} />
        </li>,
      );
    if (!out.length) out.push(<li key={`${i}-n`}>{who} asked for changes without a comment</li>);
    return out;
  });
  return (
    <section aria-label="What changed" className="rounded-xl border border-info/30 bg-info-soft/40 px-4 py-3 text-sm">
      <p className="flex items-center gap-2 font-medium">
        <History className="size-4 shrink-0 text-info" />
        {round.unchanged
          ? `The stage ran again on this request and came back with the same ${what}.`
          : `This ${what} replaces the one from ${timeAgo(round.previous_at)}.`}
      </p>
      {lines.length > 0 ? (
        <ul className="mt-1 list-disc space-y-0.5 pl-10">{lines}</ul>
      ) : (
        <p className="mt-1 pl-6 text-ink-soft">
          {round.inputs_changed.length
            ? `It was redone because earlier results changed (${round.inputs_changed.map(humanize).join(", ")}).`
            : "No comment was recorded for this round."}
        </p>
      )}
      {children && !round.unchanged && <div className="mt-2 pl-6">{children}</div>}
    </section>
  );
}
