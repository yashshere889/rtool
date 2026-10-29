import { useCodeLog } from "@/api/hooks";
import type { RunDetail } from "@/api/types";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { ErrorNote, Loading } from "@/components/ui/states";

/** The end of a log file (smoke test, package install or Slurm job). */
export function LogDialog({ run, name, title, onClose }: { run: RunDetail; name: string | null; title: string; onClose: () => void }) {
  const log = useCodeLog(run.project, run.run_id, name);
  return (
    <Dialog open={name !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent title={title} description={log.data?.truncated ? "The last part of the log." : undefined} className="w-[min(60rem,calc(100vw-2rem))]">
        {log.isPending && <Loading label="Loading the log" />}
        {log.isError && <ErrorNote error={log.error} />}
        {log.data && (
          <pre className="max-h-[65vh] overflow-auto rounded-lg bg-muted p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap">
            {log.data.text || "(empty)"}
          </pre>
        )}
      </DialogContent>
    </Dialog>
  );
}
