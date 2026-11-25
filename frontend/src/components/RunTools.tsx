import { useId, useState } from "react";
import { useNavigate } from "react-router";

import { runPath } from "@/api/client";
import { useArchive, useReplay, useStartArchive } from "@/api/hooks";
import type { RunDetail } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Field, Input } from "@/components/ui/field";
import { Spinner } from "@/components/ui/spinner";
import { count, fileSize } from "@/lib/format";
import { stageTitle } from "@/lib/stages";

/** Redo one stage in a copy of the run: the same inputs, the original untouched. */
function ReplayDialog({ run, onClose }: { run: RunDetail; onClose: () => void }) {
  const [stage, setStage] = useState(run.stages[0]?.id ?? "");
  const [tag, setTag] = useState("again");
  const replay = useReplay(run.project, run.run_id);
  const navigate = useNavigate();
  const ids = { stage: useId(), tag: useId() };
  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent
        title="Redo a stage in a copy"
        description="Copies this whole run and runs the chosen stage again in the copy, from the inputs it had the first time. This run stays as it is. The copy stops at that stage's checkpoint."
      >
        <form
          className="space-y-4"
          onSubmit={(event) => {
            event.preventDefault();
            replay.mutate({ stage, tag: tag.trim() }, { onSuccess: (made) => navigate(runPath(made.project, made.run_id)) });
          }}
        >
          <Field label="Stage to redo" htmlFor={ids.stage}>
            <select id={ids.stage} className="h-9 w-full rounded-lg border border-line bg-surface px-3 text-sm" value={stage} onChange={(e) => setStage(e.target.value)}>
              {run.stages.map((s) => (
                <option key={s.id} value={s.id}>
                  {stageTitle(s.id)}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Name for the copy" htmlFor={ids.tag} hint={`The copy is called ${run.run_id}-${stage}-${tag || "…"}.`}>
            <Input id={ids.tag} value={tag} pattern="[A-Za-z0-9][A-Za-z0-9_.\-]{0,30}" onChange={(e) => setTag(e.target.value)} />
          </Field>
          <p className="text-sm text-ink-soft">It uses the model again (and the cluster, if the stage is the code), so it costs what that stage cost.</p>
          {replay.isError && (
            <p role="alert" className="rounded-lg bg-bad-soft px-3 py-2 text-sm text-bad">
              {(replay.error as Error).message}
            </p>
          )}
          <div className="flex justify-end gap-2">
            <Button variant="ghost" type="button" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" busy={replay.isPending} disabled={!stage || !tag.trim()}>
              Make the copy
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}

/** Pack the run's results, reports, code, state and logs into one file in the archive folder. */
function ArchiveDialog({ run, onClose }: { run: RunDetail; onClose: () => void }) {
  const archive = useArchive(run.project, run.run_id, true);
  const start = useStartArchive(run.project, run.run_id);
  const [cap, setCap] = useState("200");
  const capId = useId();
  const view = archive.data;
  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent
        title="Archive this run"
        description="Packs the run's artifacts, reports, code, state and logs into one .tar.gz in the archive folder, so they outlive the scratch space. Rebuildable things (environments, caches) are left out."
      >
        {!view ? (
          <Spinner label="Looking at the run" />
        ) : (
          <div className="space-y-4 text-sm">
            {!view.can_archive ? (
              <p className="rounded-lg bg-warn-soft px-3 py-2 text-warn">Not possible here: {view.why_not}.</p>
            ) : view.state === "running" ? (
              <Spinner label="Packing the run" />
            ) : view.state === "done" ? (
              <div className="space-y-2">
                <p className="rounded-lg bg-ok-soft px-3 py-2 text-ok">
                  Archived {count(view.files, "file")} ({fileSize(view.bytes)} before compression) to <span className="font-mono break-all">{view.path}</span> on the server.
                </p>
                {view.skipped.length > 0 && (
                  <div>
                    <p className="text-warn">Left out for being larger than {cap} MB:</p>
                    <ul className="mt-1 space-y-0.5 text-xs">
                      {view.skipped.map((s) => (
                        <li key={s.path}>
                          <span className="font-mono">{s.path}</span> ({fileSize(s.bytes)})
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <>
                {view.state === "failed" && <p role="alert" className="rounded-lg bg-bad-soft px-3 py-2 text-bad">{view.error}</p>}
                <p>
                  About {fileSize(view.run_bytes)} would be packed (before compression). The archive is not downloaded; it is kept on the server, where the folder is set.
                </p>
                <Field label="Leave out single files larger than (MB)" htmlFor={capId} hint={`At most ${view.default_max_file_mb}: a limit can be lowered, not lifted.`}>
                  <Input id={capId} type="number" min={1} max={view.default_max_file_mb} value={cap} onChange={(e) => setCap(e.target.value)} />
                </Field>
              </>
            )}
            {start.isError && <p role="alert" className="text-bad">{(start.error as Error).message}</p>}
            <div className="flex justify-end gap-2">
              <Button variant="ghost" onClick={onClose}>
                Close
              </Button>
              {view.can_archive && view.state !== "running" && view.state !== "done" && (
                <Button busy={start.isPending} disabled={!(Number(cap) > 0 && Number(cap) <= view.default_max_file_mb)} onClick={() => start.mutate(Number(cap))}>
                  Archive
                </Button>
              )}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

/** Things to do with a whole run rather than with one stage. */
export function RunTools({ run }: { run: RunDetail }) {
  const [open, setOpen] = useState<"replay" | "archive" | null>(null);
  return (
    <>
      <div className="flex flex-wrap gap-2">
        <Button variant="secondary" size="sm" onClick={() => setOpen("replay")}>
          Redo a stage in a copy
        </Button>
        <Button variant="secondary" size="sm" onClick={() => setOpen("archive")}>
          Archive this run
        </Button>
      </div>
      {open === "replay" && <ReplayDialog run={run} onClose={() => setOpen(null)} />}
      {open === "archive" && <ArchiveDialog run={run} onClose={() => setOpen(null)} />}
    </>
  );
}
