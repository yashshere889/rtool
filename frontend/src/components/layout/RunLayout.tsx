import { ChevronLeft, LayoutDashboard } from "lucide-react";
import { Link, NavLink, Outlet, useOutletContext, useParams } from "react-router";

import { ApiError, runPath } from "@/api/client";
import { useRun, useRunStream } from "@/api/hooks";
import type { RunDetail } from "@/api/types";
import { cn } from "@/lib/cn";
import { stageState, stageTitle } from "@/lib/stages";

import { StageIcon } from "../StageStepper";
import { Empty, ErrorNote, Loading } from "../ui/states";

interface RunContext {
  run: RunDetail;
}

export function useRunContext(): RunContext {
  return useOutletContext<RunContext>();
}

function RunNav({ run }: { run: RunDetail }) {
  const base = runPath(run.project, run.run_id);
  const item = ({ isActive }: { isActive: boolean }) =>
    cn(
      "flex items-center gap-3 rounded-lg px-2 py-1.5 text-sm shrink-0",
      isActive ? "bg-surface shadow-sm ring-1 ring-line" : "hover:bg-muted",
    );
  return (
    <nav aria-label="Run" className="lg:sticky lg:top-20">
      <Link to="/" className="mb-3 inline-flex items-center gap-1 text-sm text-ink-soft hover:text-ink">
        <ChevronLeft className="size-4" /> All projects
      </Link>
      <p className="mb-1 line-clamp-3 text-sm font-semibold" title={run.topic}>
        {run.topic || run.run_id}
      </p>
      <p className="mb-4 text-xs text-ink-soft">
        {run.project} · {run.run_id}
      </p>
      <ul className="-mx-1 flex gap-1 overflow-x-auto px-1 pb-1 [scrollbar-width:none] lg:mx-0 lg:flex-col lg:overflow-visible lg:px-0">
        <li>
          <NavLink to={base} end className={item}>
            <span className="flex size-7 items-center justify-center">
              <LayoutDashboard className="size-4 text-ink-soft" />
            </span>
            <span className="font-medium">Overview</span>
          </NavLink>
        </li>
        {run.stages.map((stage, i) => (
          <li key={stage.id}>
            <NavLink to={`${base}/${stage.id}`} className={item}>
              <StageIcon stage={stage} run={run} index={i} />
              <span className="min-w-0">
                <span className="block font-medium">{stageTitle(stage.id)}</span>
                <span className="block truncate text-xs text-ink-soft">{stageState(stage, run).label}</span>
              </span>
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}

export function RunLayout() {
  const { project = "", runId = "" } = useParams();
  const run = useRun(project, runId);
  useRunStream(project, runId);

  if (run.isPending) return <Loading label="Loading the run" />;
  if (run.isError) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-10">
        {run.error instanceof ApiError && run.error.status === 404 ? (
          <Empty title="This run does not exist" action={<Link to="/" className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent">All projects</Link>}>
            {project}/{runId} was not found in this server's runs folder.
          </Empty>
        ) : (
          <ErrorNote error={run.error} title="Could not load the run" />
        )}
      </div>
    );
  }
  return (
    <div className="mx-auto max-w-7xl px-4 py-6 lg:grid lg:grid-cols-[15rem_minmax(0,1fr)] lg:gap-8 print:block print:p-0">
      <aside className="mb-6 lg:mb-0 print:hidden">
        <RunNav run={run.data} />
      </aside>
      <div className="min-w-0">
        <Outlet context={{ run: run.data } satisfies RunContext} />
      </div>
    </div>
  );
}
