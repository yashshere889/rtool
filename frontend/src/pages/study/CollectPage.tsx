import { Download, Upload } from "lucide-react";
import { useId, useState } from "react";

import { ApiError, fileUrl } from "@/api/client";
import { useCollect, useProtocol, useUploadData } from "@/api/hooks";
import type { RunDetail, StageSummary } from "@/api/types";
import { CheckpointCard } from "@/components/CheckpointCard";
import { FailureCard } from "@/components/FailureCard";
import { StatusBadge } from "@/components/StatusBadge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Empty, ErrorNote, Loading } from "@/components/ui/states";
import { STAGES } from "@/lib/stages";

import { Section } from "../plan/PlanRead";

function downloadTemplate(text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "data.csv";
  link.click();
  URL.revokeObjectURL(url);
}

/** The server lists every way the file breaks the protocol, joined by "; ". */
function Problems({ message }: { message: string }) {
  const [head, ...rest] = message.split(": ");
  const problems = rest.join(": ").split("; ");
  return (
    <div role="alert" className="space-y-1 rounded-lg bg-bad-soft px-4 py-3 text-sm text-bad">
      <p className="font-medium">{rest.length ? head : message}</p>
      {rest.length > 0 && (
        <ul className="list-disc space-y-0.5 pl-5">
          {problems.map((p) => (
            <li key={p}>{p}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function CollectPage({ run, stage }: { run: RunDetail; stage: StageSummary }) {
  const collect = useCollect(run.project, run.run_id);
  const protocol = useProtocol(run.project, run.run_id);
  const upload = useUploadData(run.project, run.run_id);
  const [file, setFile] = useState<File | null>(null);
  const fileId = useId();
  const header = (
    <header>
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold">Study data</h1>
        <StatusBadge stage={stage} run={run} />
      </div>
      <p className="mt-2 max-w-3xl text-sm text-ink-soft">{STAGES.collect.description}</p>
    </header>
  );
  if (collect.isPending) return <div className="space-y-6">{header}<Loading label="Loading" /></div>;
  if (collect.isError) {
    const missing = collect.error instanceof ApiError && collect.error.status === 404;
    return (
      <div className="space-y-6">
        {header}
        {missing ? <Empty title="Not yet">This stage starts once the protocol is approved.</Empty> : <ErrorNote error={collect.error} />}
      </div>
    );
  }
  const view = collect.data;
  const units = view.paired ? `${view.n_per_condition} ${view.unit}s in all` : `${view.n_per_condition} ${view.unit}s per condition`;
  const have = view.inbox_file || upload.isSuccess;
  return (
    <div className="space-y-6">
      {header}
      {stage.gate && have && <CheckpointCard run={run} gate={stage.gate} showStageLink={false} />}
      {stage.status === "failed" && <FailureCard run={run} stage={stage} />}

      <Section title="What to bring back">
        <p className="text-sm">
          Carry out the study as the protocol says, then record one row for each {view.unit} and condition: {units}.
        </p>
        <p className="text-sm text-ink-soft">{view.data_format}.</p>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          {view.columns.map((c) => (
            <Badge key={c}>{c}</Badge>
          ))}
          <Button variant="secondary" size="sm" onClick={() => downloadTemplate(view.template)}>
            <Download /> Template with the units and conditions filled in
          </Button>
          {protocol.data?.protocol.markdown_path && (
            <a className="text-accent underline underline-offset-2" href={fileUrl(run.project, run.run_id, protocol.data.protocol.markdown_path)} target="_blank" rel="noreferrer">
              Protocol document
            </a>
          )}
        </div>
      </Section>

      {view.received ? (
        <Section title="Data received">
          <p className="text-sm">
            {view.received.rows} rows read:{" "}
            {Object.entries(view.received.units)
              .map(([c, n]) => `${n} under ${c}`)
              .join(", ")}
            .
          </p>
        </Section>
      ) : (
        <Section title="Upload the data">
          {view.can_upload ? (
            <form
              className="space-y-3"
              onSubmit={(event) => {
                event.preventDefault();
                if (file) upload.mutate(file);
              }}
            >
              <label htmlFor={fileId} className="block text-sm font-medium">
                data.csv
              </label>
              <input
                id={fileId}
                type="file"
                accept=".csv,text/csv"
                className="block text-sm"
                onChange={(e) => {
                  setFile(e.target.files?.[0] ?? null);
                  upload.reset();
                }}
              />
              <p className="text-xs text-ink-soft">The file is checked against the protocol before it is kept; nothing is kept if it does not fit.</p>
              <Button type="submit" disabled={!file} busy={upload.isPending}>
                <Upload /> Upload and check
              </Button>
              {upload.isError && <Problems message={(upload.error as Error).message} />}
              {upload.isSuccess && (
                <p role="status" className="rounded-lg bg-ok-soft px-4 py-3 text-sm text-ok">
                  Received {upload.data.rows} rows
                  {stage.gate ? ". Approve below to have it analysed." : ". Retry the stage to analyse it."}
                </p>
              )}
            </form>
          ) : (
            <p className="text-sm text-ink-soft">Uploading: {view.why_not}.</p>
          )}
        </Section>
      )}
    </div>
  );
}
