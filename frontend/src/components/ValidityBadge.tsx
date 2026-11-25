import { ShieldAlert, ShieldCheck } from "lucide-react";
import { Link } from "react-router";

import { useRunValidity } from "@/api/hooks";
import type { RunDetail } from "@/api/types";

/** What this run's field has been measured at, said where its results are read (and printed with them). */
export function ValidityBadge({ run }: { run: RunDetail }) {
  const note = useRunValidity(run.project, run.run_id);
  if (!note.data) return null;
  const Icon = note.data.measured ? ShieldCheck : ShieldAlert;
  return (
    <aside
      aria-label="How far these results can be trusted"
      className="flex gap-2 rounded-lg border border-line bg-muted/50 px-3 py-2 text-sm"
    >
      <Icon className={note.data.measured ? "mt-0.5 size-4 shrink-0 text-ok" : "mt-0.5 size-4 shrink-0 text-warn"} />
      <p>
        <span className="font-medium">{note.data.measured ? "Measured field. " : "Not measured. "}</span>
        {note.data.text}{" "}
        <Link to="/settings?tab=validity" className="text-accent underline underline-offset-2 print:hidden">
          See the card
        </Link>
      </p>
    </aside>
  );
}
