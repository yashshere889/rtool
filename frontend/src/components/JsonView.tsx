import { ChevronRight } from "lucide-react";
import { useState } from "react";

import { cn } from "@/lib/cn";
import { humanize } from "@/lib/format";

/** A readable, collapsible view of any artifact (used for stages without a dedicated page). */
export function JsonView({ value, name, depth = 0 }: { value: unknown; name?: string; depth?: number }) {
  const [open, setOpen] = useState(depth < 1);
  const label =
    name === undefined ? null : (
      <span className="font-medium text-ink-soft">{/^\d+$/.test(name) ? `#${Number(name) + 1}` : humanize(name)}</span>
    );

  if (value === null || typeof value !== "object") {
    return (
      <div className="flex gap-2 py-0.5 text-sm">
        {label}
        {label && <span className="text-ink-soft">:</span>}
        <span className={cn("break-words", typeof value === "string" ? "" : "font-mono text-accent")}>
          {value === null ? "—" : typeof value === "string" ? value || "—" : String(value)}
        </span>
      </div>
    );
  }
  const entries = Array.isArray(value) ? value.map((v, i) => [String(i), v] as const) : Object.entries(value);
  if (!entries.length) {
    return (
      <div className="flex gap-2 py-0.5 text-sm">
        {label}
        <span className="text-ink-soft">(empty)</span>
      </div>
    );
  }
  return (
    <div className="text-sm">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 py-0.5 text-left hover:text-accent"
        aria-expanded={open}
      >
        <ChevronRight className={cn("size-3.5 transition-transform", open && "rotate-90")} />
        {label ?? <span className="font-medium">Contents</span>}
        <span className="text-xs text-ink-soft">
          {Array.isArray(value) ? `${entries.length} item${entries.length === 1 ? "" : "s"}` : ""}
        </span>
      </button>
      {open && (
        <div className="ml-2 border-l border-line pl-3">
          {entries.map(([key, child]) => (
            <JsonView key={key} name={key} value={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}
