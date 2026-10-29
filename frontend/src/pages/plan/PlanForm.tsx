import { ExternalLink } from "lucide-react";
import { useId, type ReactNode } from "react";
import { Link } from "react-router";

import type { ClusterView, ExperimentPlan, PlanCheck } from "@/api/types";
import { RecordList, TextList, type FieldSpec } from "@/components/ListEditor";
import { Badge } from "@/components/ui/badge";
import { Field, Input, NumberInput, Textarea } from "@/components/ui/field";
import { RESOURCE_KINDS, RESOURCE_SOURCES, resourceCheck, resourceLink, seedList } from "@/lib/plan";

import { Budget, Section } from "./PlanRead";

type Plan = ExperimentPlan;
type Item<K extends keyof Plan> = Plan[K] extends (infer T)[] ? T : never;

const VARIABLE: FieldSpec<Item<"independent_variables">>[] = [
  { key: "name", label: "Name", kind: "text" },
  { key: "values", label: "Levels (comma-separated)", kind: "csv", placeholder: "e.g. sham, phase-locked" },
  { key: "description", label: "Description", kind: "text", wide: true },
];
const MEASURED: FieldSpec<Item<"dependent_variables">>[] = [
  { key: "name", label: "Name", kind: "text" },
  { key: "description", label: "Description", kind: "text" },
];
const CONDITION: FieldSpec<Item<"conditions">>[] = [
  { key: "name", label: "Name (becomes a key in the results)", kind: "text" },
  { key: "is_baseline", label: "Baseline (the comparison point)", kind: "radio" },
  { key: "description", label: "What happens", kind: "text", wide: true },
];
const ABLATION: FieldSpec<Item<"ablations">>[] = [
  { key: "name", label: "Name", kind: "text" },
  { key: "description", label: "What is left out", kind: "text" },
];
const METRIC: FieldSpec<Item<"metrics">>[] = [
  { key: "name", label: "Name (becomes a key in the results)", kind: "text" },
  { key: "higher_is_better", label: "Higher is better", kind: "checkbox" },
  { key: "primary", label: "Primary outcome", kind: "radio" },
  { key: "definition", label: "Definition", kind: "text", wide: true },
];
const TEST: FieldSpec<Item<"statistical_tests">>[] = [
  { key: "test", label: "Test", kind: "text", placeholder: "e.g. paired t-test" },
  { key: "comparison", label: "What is compared", kind: "text" },
  { key: "alpha", label: "Significance level (alpha)", kind: "number", step: "0.01" },
  { key: "correction", label: "Correction for several tests", kind: "text", placeholder: "none, holm, bonferroni" },
];
const RESOURCE: FieldSpec<Item<"resources">>[] = [
  { key: "name", label: "Name", kind: "text" },
  { key: "kind", label: "Kind", kind: "select", options: RESOURCE_KINDS },
  { key: "source", label: "Where from", kind: "select", options: RESOURCE_SOURCES },
  { key: "identifier", label: "Identifier (repo id, address or package)", kind: "text" },
  { key: "subset", label: "Subset (optional)", kind: "text", nullable: true },
  { key: "split", label: "Split (optional)", kind: "text", nullable: true },
  { key: "purpose", label: "What it is for", kind: "text", wide: true },
];

const BLANK = {
  variable: { name: "", description: "", values: [] },
  condition: { name: "", description: "", is_baseline: false },
  metric: { name: "", definition: "", higher_is_better: true, primary: false },
  test: { test: "", comparison: "", alpha: 0.05, correction: "none" },
  resource: {
    kind: "dataset",
    name: "",
    source: "huggingface",
    identifier: "",
    subset: null,
    split: null,
    purpose: "",
    verified: false,
    license: "",
    size_gb: null,
    note: "",
  },
} as const;

function Sub({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold">{title}</h3>
      {hint && <p className="text-xs text-ink-soft">{hint}</p>}
      {children}
    </div>
  );
}

/** Every editable part of the plan. Derived values (runs, GPU-hours, results schema, issues) come from the check. */
export function PlanForm({
  value,
  base,
  onChange,
  cluster,
  checked,
  title,
  hypothesesPath,
}: {
  value: Plan;
  base: Plan; // the version being edited
  onChange: (plan: Plan) => void;
  cluster: ClusterView;
  checked: PlanCheck | undefined;
  title: string;
  hypothesesPath: string;
}) {
  const ids = { objective: useId(), design: useId(), expected: useId(), seeds: useId(), partition: useId() };
  const computeIds = { gpus: useId(), memory: useId(), hours: useId(), limit: useId(), why: useId() };
  const set = <K extends keyof Plan>(key: K, next: Plan[K]) => onChange({ ...value, [key]: next });
  const compute = value.compute;
  const setCompute = (patch: Partial<Plan["compute"]>) => set("compute", { ...compute, ...patch });
  const worked = checked?.ok ? checked.plan?.compute : undefined;
  const partitions = [...cluster.partitions].sort((a, b) => Number(b.allowed) - Number(a.allowed));
  const known = partitions.some((p) => p.name === compute.partition);
  // until the researcher sets a time limit, it follows the hours (as the server works it out)
  const autoLimit = compute.time_limit === base.compute.time_limit;
  const shownLimit = autoLimit && worked ? worked.time_limit : compute.time_limit;

  return (
    <div className="space-y-4">
      <Section title="What it tests">
        <p className="leading-relaxed">{title ? <span className="font-medium">{title}: </span> : null}{value.hypothesis}</p>
        <p className="text-sm text-ink-soft">
          The hypothesis comes from your choice; to test another one,{" "}
          <Link to={hypothesesPath} className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent">
            change it on the Hypotheses page
          </Link>
          .
        </p>
        <Field label="Objective" htmlFor={ids.objective}>
          <Textarea id={ids.objective} rows={2} className="min-h-0" value={value.objective} onChange={(e) => set("objective", e.target.value)} />
        </Field>
        <Field label="Design" hint="One paragraph: what is done to what, and how it is compared." htmlFor={ids.design}>
          <Textarea id={ids.design} rows={4} value={value.design} onChange={(e) => set("design", e.target.value)} />
        </Field>
      </Section>

      <Section title="Variables">
        <Sub title="Varied (independent)">
          <RecordList items={value.independent_variables} onChange={(v) => set("independent_variables", v)} fields={VARIABLE} blank={BLANK.variable as never} noun="independent variable" />
        </Sub>
        <Sub title="Measured (dependent)">
          <RecordList items={value.dependent_variables} onChange={(v) => set("dependent_variables", v)} fields={MEASURED} blank={BLANK.variable as never} noun="dependent variable" />
        </Sub>
        <Sub title="Kept the same (controls)">
          <TextList items={value.controls} onChange={(v) => set("controls", v)} noun="control" />
        </Sub>
      </Section>

      <Section title="Conditions">
        <Sub title="Conditions" hint="Each runs once per seed. Mark one as the baseline the others are compared with.">
          <RecordList items={value.conditions} onChange={(v) => set("conditions", v)} fields={CONDITION} blank={BLANK.condition as never} noun="condition" />
        </Sub>
        <Sub title="Ablations" hint="Variants with one part removed, to see what that part contributes (optional).">
          <RecordList items={value.ablations} onChange={(v) => set("ablations", v)} fields={ABLATION} blank={BLANK.condition as never} noun="ablation" />
        </Sub>
      </Section>

      <Section title="Measurements and statistics">
        <Sub
          title="Metrics"
          hint="The primary outcome is the one metric that decides the hypothesis: its comparisons are confirmatory, the others exploratory."
        >
          <RecordList items={value.metrics} onChange={(v) => set("metrics", v)} fields={METRIC} blank={BLANK.metric as never} noun="metric" />
        </Sub>
        <Sub title="Statistical tests">
          <RecordList items={value.statistical_tests} onChange={(v) => set("statistical_tests", v)} fields={TEST} blank={BLANK.test as never} noun="test" />
        </Sub>
        <Field label="Independent runs (seeds) per condition" hint={`The planner asks for at least ${cluster.min_seeds}.`} htmlFor={ids.seeds} className="max-w-xs">
          <NumberInput id={ids.seeds} min={1} max={50} value={value.seeds.length} onValue={(n) => set("seeds", seedList(n))} />
        </Field>
      </Section>

      <Section title="Procedure">
        <p className="text-xs text-ink-soft">Ordered steps the coding agent follows.</p>
        <TextList items={value.procedure} onChange={(v) => set("procedure", v)} noun="step" numbered multiline />
      </Section>

      <Section title="Datasets, models and software">
        <p className="text-xs text-ink-soft">
          A resource you add or point somewhere else is marked “not checked yet” until the download step fetches it.
        </p>
        <RecordList
          items={value.resources}
          onChange={(v) => set("resources", v)}
          fields={RESOURCE}
          blank={BLANK.resource as never}
          noun="resource"
          extra={(r) => {
            const link = resourceLink(r);
            const shown = checked?.ok ? checked.plan?.resources.find((x) => x.name === r.name && x.identifier === r.identifier) : undefined;
            const check = resourceCheck(shown ?? r);
            return (
              <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                <Badge tone={check.tone}>{check.label}</Badge>
                {(shown ?? r).license && <span className="text-ink-soft">licence {(shown ?? r).license}</span>}
                {(shown ?? r).size_gb != null && <span className="text-ink-soft">{(shown ?? r).size_gb} GB</span>}
                {link && (
                  <a href={link} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent">
                    Look it up <ExternalLink className="size-3" />
                  </a>
                )}
              </div>
            );
          }}
        />
      </Section>

      <Section title="Compute">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <Field label="Partition" htmlFor={ids.partition}>
            {partitions.length ? (
              <select
                id={ids.partition}
                className="h-9 w-full rounded-lg border border-line bg-surface px-3 text-sm"
                value={compute.partition}
                onChange={(e) => setCompute({ partition: e.target.value })}
              >
                {!known && <option value={compute.partition}>{compute.partition} (not known)</option>}
                {partitions.map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.name}
                    {p.gpus_per_node ? ` · ${p.gpus_per_node}× ${p.gpu_type} ${p.gpu_memory_gb ?? "?"} GB` : " · CPU"}
                    {p.allowed ? "" : " · not for experiments"}
                  </option>
                ))}
              </select>
            ) : (
              <Input id={ids.partition} value={compute.partition} onChange={(e) => setCompute({ partition: e.target.value })} />
            )}
          </Field>
          <Field label="GPUs per run" htmlFor={computeIds.gpus}>
            <NumberInput id={computeIds.gpus} min={0} max={8} value={compute.gpus_per_run} onValue={(n) => setCompute({ gpus_per_run: n })} />
          </Field>
          <Field label="GPU memory one run needs (GB)" htmlFor={computeIds.memory}>
            <NumberInput id={computeIds.memory} min={0} step="any" value={compute.gpu_memory_gb} onValue={(n) => setCompute({ gpu_memory_gb: n })} />
          </Field>
          <Field label="Hours one run takes" htmlFor={computeIds.hours}>
            <NumberInput id={computeIds.hours} min={0.05} step="any" value={compute.hours_per_run} onValue={(n) => setCompute({ hours_per_run: n })} />
          </Field>
          <Field
            label="Time limit per run"
            hint={autoLimit ? "Worked out from the hours unless you change it (HH:MM:SS)." : "Set by you (HH:MM:SS)."}
            htmlFor={computeIds.limit}
          >
            <Input id={computeIds.limit} value={shownLimit} onChange={(e) => setCompute({ time_limit: e.target.value })} />
          </Field>
        </div>
        <Field label="Why this much" htmlFor={computeIds.why}>
          <Textarea id={computeIds.why} rows={2} className="min-h-0" value={compute.justification} onChange={(e) => setCompute({ justification: e.target.value })} />
        </Field>
        {worked && (
          <div className="space-y-2 rounded-lg bg-muted px-3 py-2">
            <p className="text-sm">
              Worked out: {worked.runs} runs, time limit {worked.time_limit}
              {worked.gpus_per_run ? ` on ${worked.gpu_type ?? "GPU"}` : ""}.
            </p>
            <Budget hours={worked.gpu_hours_total} budget={cluster.budget_gpu_hours} />
          </div>
        )}
      </Section>

      <Section title="Expected results and risks">
        <Field label="Expected results" htmlFor={ids.expected}>
          <Textarea id={ids.expected} rows={3} value={value.expected_results} onChange={(e) => set("expected_results", e.target.value)} />
        </Field>
        <Sub title="Risks">
          <TextList items={value.risks} onChange={(v) => set("risks", v)} noun="risk" />
        </Sub>
      </Section>
    </div>
  );
}
