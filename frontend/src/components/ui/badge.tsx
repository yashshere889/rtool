import type { ReactNode } from "react";

import { cn } from "@/lib/cn";
import { TONE_STYLES, type Tone } from "@/lib/stages";

export function Badge({ tone = "neutral", className, children }: { tone?: Tone; className?: string; children: ReactNode }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium whitespace-nowrap",
        TONE_STYLES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
