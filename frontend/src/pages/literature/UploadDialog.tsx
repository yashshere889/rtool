import { Upload } from "lucide-react";
import { useId, useState } from "react";
import { toast } from "sonner";

import { uploadPaper } from "@/api/client";
import { useDecide } from "@/api/hooks";
import type { RunDetail, Uploaded } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Field, Input } from "@/components/ui/field";

/** Give the literature agent a paper you have (paywalled, from your library): PDF or HTML. */
export function UploadDialog({
  run,
  stage,
  open,
  onOpenChange,
  paperId,
  initialTitle = "",
}: {
  run: RunDetail;
  stage: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  paperId?: string;
  initialTitle?: string;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState(initialTitle);
  const [doi, setDoi] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [done, setDone] = useState<Uploaded | null>(null);
  const decide = useDecide(run.project, run.run_id);
  const ids = { file: useId(), title: useId(), doi: useId() };
  const gate = run.open_gates.find((g) => g.stage === stage);

  const close = () => {
    onOpenChange(false);
    setDone(null);
    setFile(null);
    setError("");
  };

  return (
    <Dialog open={open} onOpenChange={(next) => (next ? onOpenChange(true) : close())}>
      <DialogContent
        title="Add a paper you have"
        description="For papers without an open-access copy: upload the PDF (or saved HTML page) from your library."
      >
        {done ? (
          <div className="space-y-4 text-sm">
            <p>
              <span className="font-medium">{done.title}</span> is waiting in the literature agent's inbox.
            </p>
            {gate ? (
              <>
                <p className="text-ink-soft">
                  The agent reads it when the literature stage runs again. Read it now? The stage runs again with your
                  paper and comes back to this checkpoint.
                </p>
                <div className="flex justify-end gap-2">
                  <Button variant="ghost" onClick={close}>
                    Later
                  </Button>
                  <Button
                    onClick={() => {
                      decide.mutate({
                        request: {
                          kind: "reject",
                          stage: gate.stage,
                          gate: gate.gate,
                          payload: { feedback: `Read the paper I added by hand: ${done.title}` },
                        },
                        label: "Read the added paper",
                      });
                      close();
                    }}
                  >
                    Read it now
                  </Button>
                </div>
              </>
            ) : (
              <>
                <p className="text-ink-soft">The agent reads it the next time the literature stage runs.</p>
                <div className="flex justify-end">
                  <Button onClick={close}>Done</Button>
                </div>
              </>
            )}
          </div>
        ) : (
          <form
            className="space-y-4"
            onSubmit={async (event) => {
              event.preventDefault();
              if (!file) return;
              setBusy(true);
              setError("");
              try {
                const uploaded = await uploadPaper<Uploaded>(run.project, run.run_id, file, {
                  title: title.trim(),
                  paper_id: paperId,
                  doi: doi.trim() || undefined,
                });
                setDone(uploaded);
                toast.success("Paper uploaded");
              } catch (e) {
                setError((e as Error).message);
              } finally {
                setBusy(false);
              }
            }}
          >
            <Field label="File (PDF or HTML)" htmlFor={ids.file}>
              <input
                id={ids.file}
                type="file"
                accept=".pdf,.html,.htm,application/pdf,text/html"
                className="block w-full text-sm file:mr-3 file:rounded-md file:border-0 file:bg-muted file:px-3 file:py-1.5 file:text-sm file:font-medium"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </Field>
            <Field label="Title" htmlFor={ids.title}>
              <Input id={ids.title} value={title} onChange={(e) => setTitle(e.target.value)} required />
            </Field>
            {!paperId && (
              <Field label="DOI (optional)" htmlFor={ids.doi} hint="Helps match it to a paper the search already found.">
                <Input id={ids.doi} value={doi} onChange={(e) => setDoi(e.target.value)} placeholder="10.1000/xyz123" />
              </Field>
            )}
            {error && (
              <p role="alert" className="text-sm text-bad">
                {error}
              </p>
            )}
            <div className="flex justify-end gap-2">
              <Button variant="ghost" onClick={close}>
                Cancel
              </Button>
              <Button type="submit" busy={busy} disabled={!file || !title.trim()}>
                <Upload /> Upload
              </Button>
            </div>
          </form>
        )}
      </DialogContent>
    </Dialog>
  );
}
