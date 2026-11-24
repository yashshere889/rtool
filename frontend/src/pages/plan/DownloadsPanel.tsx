import { ExternalLink, Upload } from "lucide-react";
import { useId, useState } from "react";

import { useDownloads, useProvideResource } from "@/api/hooks";
import type { DownloadItem, RunDetail } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { fileSize } from "@/lib/format";
import { DOWNLOAD_STATES, resourceLink } from "@/lib/plan";
import { stageState } from "@/lib/stages";

import { Section } from "./PlanRead";

/** Put a file of a resource that is not public into its folder (small files; large data is copied on the cluster). */
function ProvideFiles({ run, item }: { run: RunDetail; item: DownloadItem }) {
  const provide = useProvideResource(run.project, run.run_id);
  const [file, setFile] = useState<File | null>(null);
  const id = useId();
  return (
    <form
      className="mt-2 flex flex-wrap items-center gap-2"
      onSubmit={(event) => {
        event.preventDefault();
        if (file) provide.mutate({ resource: item.name, file }, { onSuccess: () => setFile(null) });
      }}
    >
      <label htmlFor={id} className="text-xs font-medium">
        Add a file to {item.name}
      </label>
      <input id={id} type="file" className="text-xs" onChange={(e) => { setFile(e.target.files?.[0] ?? null); provide.reset(); }} />
      <Button type="submit" size="sm" variant="secondary" disabled={!file} busy={provide.isPending}>
        <Upload /> Upload
      </Button>
      {provide.isError && <span role="alert" className="w-full text-xs text-bad">{(provide.error as Error).message}</span>}
      {provide.isSuccess && <span role="status" className="w-full text-xs text-ok">Added {provide.data.filename} ({provide.data.files} files there now).</span>}
    </form>
  );
}

/** What the download step got for the approved plan (the fetch stage has no page of its own). */
export function DownloadsPanel({ run }: { run: RunDetail }) {
  const downloads = useDownloads(run.project, run.run_id);
  const view = downloads.data;
  if (!view?.started) return null;
  const stage = run.stages.find((s) => s.id === view.stage);
  const state = stage ? stageState(stage, run) : null;
  const done = view.items.filter((i) => i.status === "ready").length;
  return (
    <Section title="Downloads" id="downloads">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        {state && <Badge tone={state.tone}>{state.label}</Badge>}
        <span className="text-ink-soft">
          {done} of {view.items.length} downloaded for plan version {view.plan_version}
          {view.root ? ` · into ${view.root}` : ""}
        </span>
      </div>
      {view.last_error && view.status === "failed" && (
        <p className="rounded-lg bg-bad-soft px-3 py-2 text-sm text-bad">
          {view.last_error} Fix it, then use Retry on the Overview page; finished downloads are kept.
        </p>
      )}
      <ul className="divide-y divide-line">
        {view.items.map((item) => {
          const shown = DOWNLOAD_STATES[item.status];
          const link = resourceLink({ source: item.source as never, identifier: item.identifier, kind: item.kind as never });
          return (
            <li key={item.name} className="flex flex-wrap items-start justify-between gap-2 py-2.5 text-sm">
              <div className="min-w-0">
                <p className="font-medium">{item.name}</p>
                <p className="text-xs break-all text-ink-soft">
                  {link ? (
                    <a href={link} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent">
                      {item.identifier} <ExternalLink className="size-3" />
                    </a>
                  ) : (
                    item.identifier
                  )}
                  {item.path ? ` → ${item.path}` : ""}
                </p>
                {item.note && <p className="mt-0.5 text-xs text-ink-soft">{item.note}</p>}
                {view.can_upload && item.status === "manual" && <ProvideFiles run={run} item={item} />}
              </div>
              <div className="flex items-center gap-2">
                {item.bytes > 0 && (
                  <span className="text-xs text-ink-soft tabular-nums">
                    {fileSize(item.bytes)}
                    {item.files ? ` · ${item.files} files` : ""}
                  </span>
                )}
                <Badge tone={shown.tone}>{shown.label}</Badge>
              </div>
            </li>
          );
        })}
      </ul>
    </Section>
  );
}
