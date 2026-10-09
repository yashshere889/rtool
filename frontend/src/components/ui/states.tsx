import { AlertTriangle } from "lucide-react";
import type { ReactNode } from "react";

import { Spinner } from "./spinner";

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center py-16 text-ink-soft">
      <Spinner label={label} />
    </div>
  );
}

export function ErrorNote({ error, title = "Something went wrong" }: { error: unknown; title?: string }) {
  return (
    <div role="alert" className="flex gap-3 rounded-xl border border-bad/30 bg-bad-soft p-4 text-sm text-bad">
      <AlertTriangle className="mt-0.5 size-4 shrink-0" />
      <div>
        <p className="font-medium">{title}</p>
        <p className="mt-1 opacity-90">{error instanceof Error ? error.message : String(error)}</p>
      </div>
    </div>
  );
}

export function Empty({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-line px-6 py-12 text-center">
      <p className="font-medium">{title}</p>
      {children && <div className="mx-auto mt-2 max-w-md text-sm text-ink-soft">{children}</div>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}
