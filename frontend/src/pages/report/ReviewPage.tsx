import { CircleAlert, CircleCheck, CircleX } from "lucide-react";
import { useSearchParams } from "react-router";

import { ApiError } from "@/api/client";
import { useReview } from "@/api/hooks";
import type { ReviewView, RunDetail, StageSummary } from "@/api/types";
import { CheckpointCard } from "@/components/CheckpointCard";
import { DiffText } from "@/components/DiffText";
import { FailureCard } from "@/components/FailureCard";
import { StatusBadge } from "@/components/StatusBadge";
import { Badge } from "@/components/ui/badge";
import { Empty, ErrorNote, Loading } from "@/components/ui/states";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/cn";
import { CRITERIA } from "@/lib/report";
import { STAGES } from "@/lib/stages";

import { Section } from "../plan/PlanRead";
import { ReportWorkspace } from "./ReportWorkspace";

type Issue = ReviewView["review"]["issues"][number];

function issueState(issue: Issue, view: ReviewView): { label: string; tone: "ok" | "info" | "warn" } {
  const review = view.review;
  if (issue.target === "report" && review.edited_sections.includes(issue.section)) return { label: "Edited by the reviewer", tone: "ok" };
  if (review.change_requests.some((r) => r.includes(issue.problem))) return { label: `Sent back (${issue.target.replace("_", " ")})`, tone: "info" };
  return { label: "Open: yours to decide", tone: "warn" };
}

function Scores({ view }: { view: ReviewView }) {
  const scores = Object.entries(view.review.scores);
  if (!scores.length) return <p className="text-sm text-ink-soft">The automatic review did not produce scores.</p>;
  return (
    <dl className="space-y-2">
      {scores.map(([name, value]) => {
        const info = CRITERIA[name] ?? { label: name, help: "" };
        const low = value < view.min_score;
        return (
          <div key={name} className="grid grid-cols-[8rem_1fr_3rem] items-center gap-3 text-sm">
            <dt>
              <Tooltip content={info.help}>
                <span className="cursor-help underline decoration-line decoration-dotted underline-offset-4">{info.label}</span>
              </Tooltip>
            </dt>
            <dd className="relative h-2.5 rounded-full bg-muted" aria-label={`${value} out of 5`}>
              <span className={cn("absolute inset-y-0 left-0 rounded-full", low ? "bg-warn" : "bg-ok")} style={{ width: `${(value / 5) * 100}%` }} />
              <span className="absolute -top-1 -bottom-1 w-0.5 bg-ink-soft/60" style={{ left: `${(view.min_score / 5) * 100}%` }} title={`needed to accept: ${view.min_score}`} />
            </dd>
            <dd className={cn("text-right tabular-nums", low && "font-semibold text-warn")}>{value}/5</dd>
          </div>
        );
      })}
      <p className="text-xs text-ink-soft">Median of independent reviews; the line marks the score needed to accept ({view.min_score}).</p>
    </dl>
  );
}

function ReviewTab({ view, goToSection }: { view: ReviewView; goToSection: (id: string) => void }) {
  const review = view.review;
  const accepted = review.verdict === "accept";
  return (
    <div className="space-y-4">
      <section
        aria-label="Verdict"
        className={cn("rounded-xl border p-4 sm:p-5", accepted ? "border-ok/40 bg-ok-soft/40" : "border-warn/40 bg-warn-soft/40")}
      >
        <p className={cn("text-xs font-semibold tracking-wide uppercase", accepted ? "text-ok" : "text-warn")}>Verdict</p>
        <p className="mt-1 text-lg font-semibold">{accepted ? "The reviewer recommends accepting the report" : "The report needs your judgment"}</p>
        <p className="mt-1 text-sm leading-relaxed">{review.summary}</p>
        {view.final_edited && (
          <p className="mt-2 text-sm text-ink-soft">You edited the final report after this review; your edit was not reviewed again.</p>
        )}
      </section>

      <div className="grid items-start gap-4 lg:grid-cols-2">
        <Section title="Scores">
          <Scores view={view} />
        </Section>
        <Section title={`Automated checks (${review.checks.filter((c) => c.ok).length} of ${review.checks.length} passed)`}>
          <ul className="space-y-1.5 text-sm">
            {review.checks.map((c) => (
              <li key={c.name} className="flex gap-2">
                {c.ok ? <CircleCheck className="mt-0.5 size-4 shrink-0 text-ok" /> : <CircleX className="mt-0.5 size-4 shrink-0 text-bad" />}
                <span>
                  <span className={cn(!c.ok && "font-medium")}>{c.name}</span>
                  {c.detail && <span className="block text-xs text-ink-soft">{c.detail}</span>}
                </span>
              </li>
            ))}
          </ul>
        </Section>
      </div>

      <Section title={`Issues raised by most reviews (${review.issues.length})`}>
        {review.issues.length === 0 ? (
          <p className="text-sm text-ink-soft">None.</p>
        ) : (
          <ul className="space-y-3">
            {review.issues.map((issue, i) => {
              const state = issueState(issue, view);
              return (
                <li key={i} className="rounded-lg border border-line bg-canvas/40 p-3 text-sm">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge tone={issue.severity === "major" ? "bad" : "neutral"}>{issue.severity}</Badge>
                    <button type="button" className="font-medium text-accent hover:underline" onClick={() => goToSection(issue.section)}>
                      {issue.section.replace(/_/g, " ")}
                    </button>
                    <Badge tone={state.tone}>{state.label}</Badge>
                  </div>
                  <p className="mt-1.5">{issue.problem}</p>
                  <p className="mt-0.5 text-ink-soft">Suggestion: {issue.suggestion}</p>
                </li>
              );
            })}
          </ul>
        )}
      </Section>

      <Section title="Sent back for revision">
        <p className="text-sm">
          The report has had {view.rounds_used} of at most {view.rounds_max} revision rounds.
        </p>
        {review.change_requests.length > 0 ? (
          <ul className="list-disc space-y-0.5 pl-5 text-sm">
            {review.change_requests.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-ink-soft">Nothing was sent back to an earlier stage in this review.</p>
        )}
      </Section>
    </div>
  );
}

export function ReviewPage({ run, stage }: { run: RunDetail; stage: StageSummary }) {
  const review = useReview(run.project, run.run_id);
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") ?? "review";
  const setTab = (value: string) =>
    setParams((prev) => {
      const next = new URLSearchParams(prev);
      next.set("tab", value);
      next.delete("mode");
      return next;
    });
  const goToSection = (id: string) => {
    setTab("final");
    // the final report may still be loading: look for the section for a few seconds
    let tries = 0;
    const find = () => {
      const target = document.getElementById(`section-${id}`);
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
      else if (++tries < 30) setTimeout(find, 150);
    };
    setTimeout(find, 50);
  };

  const header = (
    <header className="print:hidden">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold">Review</h1>
        <StatusBadge stage={stage} run={run} />
      </div>
      <p className="mt-2 max-w-3xl text-sm text-ink-soft">{STAGES.review.description}</p>
    </header>
  );
  const checkpoint = stage.gate && (
    <CheckpointCard
      run={run}
      gate={stage.gate}
      showStageLink={false}
      rejectHint="Your comments go to the writer; sections you wrote are kept. The new report is then reviewed again."
    />
  );

  if (review.isPending) return <div className="space-y-6">{header}<Loading label="Loading the review" /></div>;
  if (review.isError) {
    const missing = review.error instanceof ApiError && review.error.status === 404;
    return (
      <div className="space-y-6">
        {header}
        {stage.status === "failed" && <FailureCard run={run} stage={stage} />}
        {missing ? <Empty title="No review yet">It appears here once the report is written and reviewed.</Empty> : <ErrorNote error={review.error} />}
      </div>
    );
  }
  const view = review.data;
  const sentBack = view.review.summary.startsWith("Sent back to the Writer");
  return (
    <div className="space-y-6">
      {header}
      <div className="print:hidden">{checkpoint}</div>
      {stage.status === "failed" && <FailureCard run={run} stage={stage} />}
      {stage.status === "stale" && stage.stale_reason && (
        <p className="rounded-lg bg-stale-soft px-4 py-3 text-sm text-stale print:hidden">Will run again: {stage.stale_reason}</p>
      )}
      {sentBack && (
        <p className="flex gap-2 rounded-lg bg-info-soft px-4 py-3 text-sm text-info print:hidden">
          <CircleAlert className="mt-0.5 size-4 shrink-0" /> {view.review.summary}
        </p>
      )}
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="print:hidden">
          <TabsTrigger value="review">Review</TabsTrigger>
          <TabsTrigger value="final">Final report</TabsTrigger>
          <TabsTrigger value="edits">Reviewer's edits ({view.review.edited_sections.length})</TabsTrigger>
        </TabsList>
        <TabsContent value="review" className="mt-4">
          <ReviewTab view={view} goToSection={goToSection} />
        </TabsContent>
        <TabsContent value="final" className="mt-4">
          <ReportWorkspace run={run} which="final_report" stage={stage.id} />
        </TabsContent>
        <TabsContent value="edits" className="mt-4 space-y-3">
          {view.diff ? (
            <>
              <p className="text-sm text-ink-soft">
                What the reviewer changed in the report ({view.review.edited_sections.map((s) => s.replace(/_/g, " ")).join(", ")}).
              </p>
              <DiffText text={view.diff} />
            </>
          ) : (
            <Empty title="No edits">The reviewer did not change the text.</Empty>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
