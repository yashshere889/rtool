import { useEvents } from "@/api/hooks";
import { cn } from "@/lib/cn";
import { describeEvent } from "@/lib/events";
import { dateTime, timeAgo } from "@/lib/format";
import { TONE_DOT } from "@/lib/stages";

import { ErrorNote, Loading } from "./ui/states";

/** What happened in the run, newest first, in plain words (kept live by the run's event stream). */
export function ActivityFeed({
  project,
  runId,
  stage,
  limit = 40,
}: {
  project: string;
  runId: string;
  stage?: string;
  limit?: number;
}) {
  const events = useEvents(project, runId);
  if (events.isPending) return <Loading label="Loading activity" />;
  if (events.isError) return <ErrorNote error={events.error} />;
  const items = events.data.events
    .filter((e) => !stage || e.stage === stage)
    .map((e) => ({ event: e, described: describeEvent(e) }))
    .filter((item) => item.described !== null)
    .reverse()
    .slice(0, limit);
  if (!items.length) return <p className="py-6 text-center text-sm text-ink-soft">Nothing has happened yet.</p>;
  return (
    <ol className="space-y-3" aria-label="Activity">
      {items.map(({ event, described }, i) => (
        <li key={`${event.ts}-${i}`} className="flex gap-3 text-sm">
          <span className={cn("mt-1.5 size-2 shrink-0 rounded-full", TONE_DOT[described!.tone])} aria-hidden />
          <div className="min-w-0 flex-1">
            <p className="break-words">{described!.text}</p>
            <time dateTime={event.ts} title={dateTime(event.ts)} className="text-xs text-ink-soft">
              {timeAgo(event.ts)}
            </time>
          </div>
        </li>
      ))}
    </ol>
  );
}
