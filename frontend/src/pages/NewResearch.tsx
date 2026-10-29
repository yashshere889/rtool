import { ChevronRight, Sparkles } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router";
import { toast } from "sonner";

import { runPath } from "@/api/client";
import { useCreateRun, useSettings } from "@/api/hooks";
import type { NewRun } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Card, CardBody } from "@/components/ui/card";
import { Field, Input, Textarea } from "@/components/ui/field";
import { ErrorNote } from "@/components/ui/states";
import { cn } from "@/lib/cn";
import { lines, projectName } from "@/lib/project";

const EXAMPLES = [
  "Does sound played during deep sleep improve memory in older adults?",
  "How robust are sentiment classifiers to label noise in the training data?",
  "Can graph neural networks predict ocean heat transport from sparse observations?",
];

const REVIEW_MODES: { value: NewRun["review"]; title: string; text: string }[] = [
  {
    value: "every_step",
    title: "Check with me at every step",
    text: "You review the papers, choose the hypothesis, approve the plan, the GPU time and the final report.",
  },
  {
    value: "gpu_and_final",
    title: "Only before GPU time and at the end",
    text: "Papers, hypothesis and plan go ahead automatically; you approve GPU spending and the final report.",
  },
  {
    value: "automatic",
    title: "Fully automatic",
    text: "Runs to the end without stopping, including GPU time. Best for small, cheap studies.",
  },
];

const PACKS = [
  { value: "generic", title: "Any field" },
  { value: "ml_dl", title: "Machine learning / deep learning (PyTorch)" },
];

export function NewResearch() {
  const navigate = useNavigate();
  const settings = useSettings();
  const create = useCreateRun();
  const [topic, setTopic] = useState("");
  const [project, setProject] = useState("");
  const [questions, setQuestions] = useState("");
  const [constraints, setConstraints] = useState("");
  const [queries, setQueries] = useState("");
  const [pack, setPack] = useState("generic");
  const [review, setReview] = useState<NewRun["review"]>("every_step");
  const [gpuHours, setGpuHours] = useState("");
  const [more, setMore] = useState(false);

  const submit = async () => {
    const created = await create.mutateAsync({
      topic: topic.trim(),
      project: project.trim() || null,
      questions: lines(questions),
      constraints: lines(constraints),
      seed_queries: lines(queries),
      pack,
      review,
      max_gpu_hours: gpuHours ? Number(gpuHours) : null,
    });
    toast.success("Research started", { description: created.kick.detail });
    navigate(runPath(created.run.project, created.run.run_id));
  };

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <h1 className="text-2xl font-semibold">New research</h1>
      <p className="mt-1 text-sm text-ink-soft">
        Describe what you want to find out. The tool searches and reads the literature first, then proposes
        hypotheses for you to choose from.
      </p>
      {settings.data?.runs_locally && (
        <p className="mt-4 rounded-lg bg-info-soft px-4 py-3 text-sm text-info">
          Demo mode: whatever you type, this computer simulates a study on sleep and memory with made-up papers.
        </p>
      )}

      <form
        className="mt-6 space-y-6"
        onSubmit={(event) => {
          event.preventDefault();
          void submit();
        }}
      >
        <Card>
          <CardBody className="space-y-4 py-5">
            <Field label="Research question or topic" htmlFor="topic" hint="One or two sentences is enough.">
              <Textarea
                id="topic"
                className="min-h-28 text-base"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="What do you want to find out?"
                required
                minLength={3}
                autoFocus
              />
            </Field>
            <div className="flex flex-wrap gap-2" role="group" aria-label="Examples">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  onClick={() => setTopic(example)}
                  className="inline-flex items-center gap-1.5 rounded-full border border-line px-3 py-1 text-xs text-ink-soft hover:border-accent/50 hover:text-ink"
                >
                  <Sparkles className="size-3" /> {example}
                </button>
              ))}
            </div>
          </CardBody>
        </Card>

        <fieldset>
          <legend className="mb-2 text-sm font-medium">How often should the tool check with you?</legend>
          <div className="grid gap-2">
            {REVIEW_MODES.map((mode) => (
              <label
                key={mode.value}
                className={cn(
                  "flex cursor-pointer gap-3 rounded-xl border bg-surface p-4",
                  review === mode.value ? "border-accent ring-2 ring-accent/20" : "border-line hover:border-ink-soft/40",
                )}
              >
                <input
                  type="radio"
                  name="review"
                  className="mt-1 accent-[var(--accent)]"
                  checked={review === mode.value}
                  onChange={() => setReview(mode.value)}
                />
                <span>
                  <span className="block font-medium">
                    {mode.title}
                    {mode.value === "every_step" && <span className="ml-2 text-xs text-accent">recommended</span>}
                  </span>
                  <span className="mt-0.5 block text-sm text-ink-soft">{mode.text}</span>
                </span>
              </label>
            ))}
          </div>
        </fieldset>

        <div>
          <button
            type="button"
            className="inline-flex items-center gap-1 text-sm font-medium text-accent"
            aria-expanded={more}
            onClick={() => setMore(!more)}
          >
            <ChevronRight className={cn("size-4 transition-transform", more && "rotate-90")} /> More options
          </button>
          {more && (
            <div className="mt-4 grid gap-4 sm:grid-cols-2">
              <Field label="Specific questions" htmlFor="questions" hint="One per line." className="sm:col-span-2">
                <Textarea id="questions" className="min-h-20" value={questions} onChange={(e) => setQuestions(e.target.value)} />
              </Field>
              <Field label="Constraints" htmlFor="constraints" hint="One per line, e.g. “public data only”.">
                <Textarea id="constraints" className="min-h-20" value={constraints} onChange={(e) => setConstraints(e.target.value)} />
              </Field>
              <Field label="Search queries to start from" htmlFor="queries" hint="One per line (optional).">
                <Textarea id="queries" className="min-h-20" value={queries} onChange={(e) => setQueries(e.target.value)} />
              </Field>
              <Field label="Field" htmlFor="pack" hint="Gives the coding agent the right starting point.">
                <select
                  id="pack"
                  className="h-9 w-full rounded-lg border border-line bg-surface px-3 text-sm"
                  value={pack}
                  onChange={(e) => setPack(e.target.value)}
                >
                  {PACKS.map((p) => (
                    <option key={p.value} value={p.value}>
                      {p.title}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="GPU-hour limit" htmlFor="gpu" hint="Optional cap for the whole study.">
                <Input id="gpu" type="number" min={1} step={1} value={gpuHours} onChange={(e) => setGpuHours(e.target.value)} />
              </Field>
              <Field
                label="Project"
                htmlFor="project"
                hint={`Runs are grouped by project. Leave empty to use “${projectName(topic)}”.`}
                className="sm:col-span-2"
              >
                <Input
                  id="project"
                  value={project}
                  pattern="[A-Za-z0-9][A-Za-z0-9_.\-]*"
                  title="Letters, digits, dots, dashes and underscores"
                  onChange={(e) => setProject(e.target.value)}
                />
              </Field>
            </div>
          )}
        </div>

        {create.isError && <ErrorNote error={create.error} title="Could not start the research" />}
        <div className="flex justify-end">
          <Button type="submit" size="lg" busy={create.isPending} disabled={topic.trim().length < 3}>
            Start research
          </Button>
        </div>
      </form>
    </div>
  );
}
