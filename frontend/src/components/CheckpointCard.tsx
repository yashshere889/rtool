import { ArrowRight, FileText, Undo2 } from "lucide-react";
import { useId, useState } from "react";
import { Link } from "react-router";

import { fileUrl, runPath } from "@/api/client";
import { useDecide, useUndo } from "@/api/hooks";
import type { ActionRequest, OpenGate, RunDetail } from "@/api/types";
import { cn } from "@/lib/cn";
import { humanize } from "@/lib/format";
import { gateInfo, stageTitle } from "@/lib/stages";

import { Button } from "./ui/button";
import { Dialog, DialogClose, DialogContent } from "./ui/dialog";
import { Textarea } from "./ui/field";

type Summary = Record<string, unknown>;

const SHOWN_ELSEWHERE = new Set([
  "prompt",
  "message",
  "artifacts",
  "options",
  "review_files",
  "requests",
  "pending_revisions",
  "reopened",
  "current",
  "tested_sha",
]);

// labels for facts agents put in a checkpoint summary (anything else is shown humanised)
const LABELS: Record<string, string> = {
  gpu_hours: "GPU-hours (at most)",
  time_limit: "Time limit per run",
  smoke: "Smoke tests",
};

interface Option {
  index: number;
  title: string;
}

interface Request {
  from_stage: string;
  target_stage: string;
  reason: string;
}

/** Decisions at one open checkpoint, bound to the gate as the page shows it (a stale page cannot decide). */
export function useGateDecision(run: RunDetail, gate: OpenGate) {
  const decide = useDecide(run.project, run.run_id);
  const undo = useUndo(run.project, run.run_id);
  const summary = gate.summary as Summary;
  const expect = { gate_nonce: gate.nonce, artifacts: (summary.artifacts as Record<string, number>) ?? {} };
  const queued = run.queued_actions.find((a) => a.stage === gate.stage && a.gate === gate.gate);
  function send(
    kind: ActionRequest["kind"],
    payload: Record<string, unknown>,
    label: string,
    options?: { onSuccess?: () => void },
  ) {
    decide.mutate(
      { request: { kind, stage: gate.stage, gate: gate.gate, payload: { ...payload, expect } }, label },
      { onSuccess: () => options?.onSuccess?.() },
    );
  }
  return { send, queued, decide, undo };
}

/** A checkpoint (gate) waiting for the researcher: what to look at, and the decision buttons.
 *
 * ``choiceElsewhere``: the page shows the options with their own buttons (e.g. the hypotheses), so the card
 * leaves out its option list. */
export function CheckpointCard({
  run,
  gate,
  showStageLink = true,
  choiceElsewhere = false,
  rejectHint,
}: {
  run: RunDetail;
  gate: OpenGate;
  showStageLink?: boolean;
  choiceElsewhere?: boolean;
  rejectHint?: string; // said in the "request changes" dialog, e.g. that hand edits are not kept
}) {
  const info = gateInfo(gate.gate);
  const summary = gate.summary as Summary;
  const { send, queued, decide, undo } = useGateDecision(run, gate);
  const [choice, setChoice] = useState<number | null>(null);
  const [feedback, setFeedback] = useState("");
  const [asking, setAsking] = useState(false);
  const feedbackId = useId();

  const options = (summary.options as Option[] | undefined) ?? [];
  const requests = (summary.requests as Request[] | undefined) ?? [];
  const files = (summary.review_files as string[] | undefined) ?? [];
  const text = String(summary.prompt ?? summary.message ?? "");
  const details = Object.entries(summary).filter(([key, value]) => !SHOWN_ELSEWHERE.has(key) && value !== "");
  const reopened = Boolean(summary.reopened);
  const current = (summary.current as { index: number } | undefined)?.index;
  const canKeep = gate.kind === "select" && gate.actions.includes("approve");
  const showOptions = gate.kind === "select" && options.length > 0 && !choiceElsewhere;

  const approve = () =>
    gate.kind === "select"
      ? send("select", { index: choice }, `${info.title}: option ${(choice ?? 0) + 1}`)
      : send("approve", {}, info.title);

  return (
    <section
      aria-label={info.title}
      className="rounded-xl border border-warn/40 bg-warn-soft/40 p-5 dark:bg-warn-soft/30"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs font-semibold tracking-wide text-warn uppercase">
            Your decision · {stageTitle(gate.stage)}
          </p>
          <h3 className="mt-1 text-lg font-semibold">{info.title}</h3>
        </div>
        {showStageLink && (
          <Link
            to={`${runPath(run.project, run.run_id)}/${gate.stage}`}
            className="inline-flex items-center gap-1 text-sm font-medium text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent"
          >
            Review on the {stageTitle(gate.stage)} page <ArrowRight className="size-4" />
          </Link>
        )}
      </div>

      {reopened && (
        <p className="mt-3 rounded-lg bg-surface/70 px-3 py-2 text-sm">
          You reopened this checkpoint.{" "}
          {canKeep
            ? `Keep your current choice${current !== undefined ? ` (option ${current + 1})` : ""}, choose another, or ask for new options.`
            : "Approving keeps the current results; requesting changes runs this stage again."}{" "}
          Later stages only run again if the outcome changes.
        </p>
      )}
      {text && <p className="mt-3 text-sm leading-relaxed">{text}</p>}
      <p className="mt-2 text-sm text-ink-soft">{choiceElsewhere ? `${info.help} Choose below.` : info.help}</p>

      {details.length > 0 && (
        <dl className="mt-3 grid gap-x-4 gap-y-1 text-sm sm:grid-cols-[max-content_1fr]">
          {details.map(([key, value]) => (
            <div key={key} className="contents">
              <dt className="font-medium text-ink-soft">{LABELS[key] ?? humanize(key)}</dt>
              <dd className="break-words">{typeof value === "string" ? value : JSON.stringify(value)}</dd>
            </div>
          ))}
        </dl>
      )}

      {requests.length > 0 && (
        <ul className="mt-3 space-y-1 text-sm">
          {requests.map((r, i) => (
            <li key={i}>
              <span className="font-medium">{stageTitle(r.from_stage)}</span> asked{" "}
              <span className="font-medium">{stageTitle(r.target_stage)}</span>: {r.reason}
            </li>
          ))}
        </ul>
      )}

      {files.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {files.map((path) => (
            <a
              key={path}
              href={fileUrl(run.project, run.run_id, path)}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 rounded-md border border-line bg-surface px-2.5 py-1 text-sm hover:bg-muted"
            >
              <FileText className="size-4 text-ink-soft" /> {path.split("/").pop()}
            </a>
          ))}
        </div>
      )}

      {showOptions && (
        <fieldset className="mt-4 space-y-2" disabled={Boolean(queued)}>
          <legend className="mb-1 text-sm font-medium">Options</legend>
          {options.map((option) => (
            <label
              key={option.index}
              className={cn(
                "flex cursor-pointer items-start gap-3 rounded-lg border bg-surface p-3 text-sm",
                choice === option.index ? "border-accent ring-2 ring-accent/20" : "border-line hover:border-ink-soft/40",
              )}
            >
              <input
                type="radio"
                name={`choice-${gate.nonce}`}
                className="mt-0.5 accent-[var(--accent)]"
                checked={choice === option.index}
                onChange={() => setChoice(option.index)}
              />
              <span>
                <span className="font-medium">Option {option.index + 1}.</span> {option.title}
                {option.index === current && <span className="text-ink-soft"> (your current choice)</span>}
              </span>
            </label>
          ))}
        </fieldset>
      )}

      {queued ? (
        <div className="mt-4 flex flex-wrap items-center gap-3 rounded-lg bg-surface/70 px-3 py-2 text-sm">
          <span>
            Your decision (<span className="font-medium">{queued.kind}</span>) is on its way; it will be applied when a
            worker picks it up.
          </span>
          <Button variant="secondary" size="sm" busy={undo.isPending} onClick={() => undo.mutate(queued.id)}>
            <Undo2 /> Undo
          </Button>
        </div>
      ) : (
        <div className="mt-4 flex flex-wrap gap-2">
          {(gate.kind !== "select" || showOptions) && (
            <Button onClick={approve} busy={decide.isPending} disabled={gate.kind === "select" && choice === null}>
              {info.approve}
            </Button>
          )}
          {canKeep && (
            <Button
              variant={showOptions ? "secondary" : "primary"}
              onClick={() => send("approve", {}, `${info.title}: keep the current choice`)}
              disabled={decide.isPending}
            >
              Keep my current choice
            </Button>
          )}
          <Button variant="secondary" onClick={() => setAsking(true)} disabled={decide.isPending}>
            {info.reject}
          </Button>
        </div>
      )}

      <Dialog open={asking} onOpenChange={setAsking}>
        <DialogContent
          title={info.reject}
          description={`The stage runs again with your comments. Be specific.${rejectHint ? ` ${rejectHint}` : ""}`}
        >
          <form
            onSubmit={(event) => {
              event.preventDefault();
              send("reject", { feedback: feedback.trim() }, `${info.title}: ${info.reject.toLowerCase()}`);
              setAsking(false);
              setFeedback("");
            }}
          >
            <label htmlFor={feedbackId} className="text-sm font-medium">
              What should change?
            </label>
            <Textarea
              id={feedbackId}
              className="mt-1.5"
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              placeholder="For example: focus on studies in older adults, and drop the modelling papers."
              autoFocus
            />
            <div className="mt-4 flex justify-end gap-2">
              <DialogClose asChild>
                <Button variant="ghost">Cancel</Button>
              </DialogClose>
              <Button type="submit" disabled={!feedback.trim()}>
                Send
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </section>
  );
}
