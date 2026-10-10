import { ArrowDown, ArrowUp, Plus, Trash2 } from "lucide-react";
import { useEffect, useId, useState, type ReactNode } from "react";

import { cn } from "@/lib/cn";

import { Button } from "./ui/button";
import { Input, NumberInput, Textarea } from "./ui/field";

export interface FieldSpec<T> {
  key: keyof T & string;
  label: string;
  kind: "text" | "textarea" | "number" | "select" | "csv" | "radio" | "checkbox";
  options?: [string, string][];
  wide?: boolean; // takes the whole row
  nullable?: boolean; // an empty text box means "none" (null), not ""
  step?: string;
  placeholder?: string;
}

function move<T>(items: T[], from: number, to: number): T[] {
  const next = [...items];
  const [item] = next.splice(from, 1);
  next.splice(to, 0, item);
  return next;
}

function ItemTools({ index, count, label, onMove, onRemove }: {
  index: number;
  count: number;
  label: string;
  onMove: (to: number) => void;
  onRemove: () => void;
}) {
  return (
    <div className="flex shrink-0 items-center gap-0.5">
      <Button variant="ghost" size="icon" className="size-7" aria-label={`Move ${label} up`} disabled={index === 0} onClick={() => onMove(index - 1)}>
        <ArrowUp />
      </Button>
      <Button variant="ghost" size="icon" className="size-7" aria-label={`Move ${label} down`} disabled={index === count - 1} onClick={() => onMove(index + 1)}>
        <ArrowDown />
      </Button>
      <Button variant="ghost" size="icon" className="size-7 hover:text-bad" aria-label={`Remove ${label}`} onClick={onRemove}>
        <Trash2 />
      </Button>
    </div>
  );
}

/** A comma-separated list in one text box (keeps what is typed, e.g. a trailing comma, while editing). */
function CsvInput({ id, value, onChange, placeholder }: { id: string; value: string[]; onChange: (v: string[]) => void; placeholder?: string }) {
  const parse = (text: string) => text.split(",").map((part) => part.trim()).filter(Boolean);
  const [text, setText] = useState(value.join(", "));
  useEffect(() => {
    if (JSON.stringify(parse(text)) !== JSON.stringify(value)) setText(value.join(", "));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only follow changes made elsewhere
  }, [value]);
  return (
    <Input
      id={id}
      value={text}
      placeholder={placeholder}
      onChange={(e) => {
        setText(e.target.value);
        onChange(parse(e.target.value));
      }}
    />
  );
}

function FieldInput<T extends object>({ field, item, items, index, onChange, id }: {
  field: FieldSpec<T>;
  item: T;
  items: T[];
  index: number;
  onChange: (items: T[]) => void;
  id: string;
}) {
  const value = item[field.key] as unknown;
  const set = (next: unknown) => onChange(items.map((it, i) => (i === index ? { ...it, [field.key]: next } : it)));
  switch (field.kind) {
    case "textarea":
      return <Textarea id={id} rows={2} className="min-h-0" value={String(value ?? "")} placeholder={field.placeholder} onChange={(e) => set(e.target.value)} />;
    case "number":
      return <NumberInput id={id} step={field.step ?? "any"} value={(value as number | null) ?? null} onValue={set} />;
    case "select":
      return (
        <select id={id} className="h-9 w-full rounded-lg border border-line bg-surface px-3 text-sm" value={String(value ?? "")} onChange={(e) => set(e.target.value)}>
          {field.options?.map(([v, label]) => (
            <option key={v} value={v}>
              {label}
            </option>
          ))}
        </select>
      );
    case "csv":
      return <CsvInput id={id} value={(value as string[]) ?? []} onChange={set} placeholder={field.placeholder} />;
    case "checkbox":
      return <input id={id} type="checkbox" className="size-4 accent-[var(--accent)]" checked={Boolean(value)} onChange={(e) => set(e.target.checked)} />;
    case "radio": // one item in the list: choosing it clears the others
      return (
        <input
          id={id}
          type="radio"
          className="size-4 accent-[var(--accent)]"
          checked={Boolean(value)}
          onChange={() => onChange(items.map((it, i) => ({ ...it, [field.key]: i === index })))}
        />
      );
    default:
      return (
        <Input
          id={id}
          value={String(value ?? "")}
          placeholder={field.placeholder}
          onChange={(e) => set(field.nullable && !e.target.value ? null : e.target.value)}
        />
      );
  }
}

/** Edit a list of records (conditions, metrics, resources ...): fields per item, add, remove, reorder. */
export function RecordList<T extends object>({
  items,
  onChange,
  fields,
  blank,
  noun,
  extra,
}: {
  items: T[];
  onChange: (items: T[]) => void;
  fields: FieldSpec<T>[];
  blank: T;
  noun: string; // "condition", used in button and screen-reader labels
  extra?: (item: T, index: number) => ReactNode;
}) {
  const base = useId();
  return (
    <div className="space-y-2">
      {items.map((item, index) => {
        const label = `${noun} ${index + 1}`;
        return (
          <div key={index} role="group" aria-label={label} className="rounded-lg border border-line bg-canvas/40 p-3">
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="text-xs font-semibold tracking-wide text-ink-soft uppercase">{label}</span>
              <ItemTools
                index={index}
                count={items.length}
                label={label}
                onMove={(to) => onChange(move(items, index, to))}
                onRemove={() => onChange(items.filter((_, i) => i !== index))}
              />
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              {fields.map((field) => {
                const id = `${base}-${index}-${field.key}`;
                const inline = field.kind === "checkbox" || field.kind === "radio";
                return (
                  <div key={field.key} className={cn(field.wide && "sm:col-span-2", inline && "flex items-center gap-2 self-end pb-2")}>
                    {!inline && (
                      <label htmlFor={id} className="mb-1 block text-xs font-medium text-ink-soft">
                        {field.label}
                      </label>
                    )}
                    <FieldInput field={field} item={item} items={items} index={index} onChange={onChange} id={id} />
                    {inline && (
                      <label htmlFor={id} className="text-sm">
                        {field.label}
                      </label>
                    )}
                  </div>
                );
              })}
            </div>
            {extra?.(item, index)}
          </div>
        );
      })}
      <Button variant="secondary" size="sm" onClick={() => onChange([...items, structuredClone(blank)])}>
        <Plus /> Add {noun}
      </Button>
    </div>
  );
}

/** Edit a list of short texts (controls, risks, procedure steps): add, remove, reorder. */
export function TextList({
  items,
  onChange,
  noun,
  numbered = false,
  multiline = false,
}: {
  items: string[];
  onChange: (items: string[]) => void;
  noun: string;
  numbered?: boolean;
  multiline?: boolean;
}) {
  const base = useId();
  return (
    <div className="space-y-2">
      {items.map((text, index) => {
        const label = `${noun} ${index + 1}`;
        const set = (value: string) => onChange(items.map((t, i) => (i === index ? value : t)));
        return (
          <div key={index} className="flex items-start gap-2">
            {numbered && <span className="mt-2 w-6 shrink-0 text-right text-sm text-ink-soft tabular-nums">{index + 1}.</span>}
            {multiline ? (
              <Textarea id={`${base}-${index}`} aria-label={label} rows={2} className="min-h-0 flex-1" value={text} onChange={(e) => set(e.target.value)} />
            ) : (
              <Input id={`${base}-${index}`} aria-label={label} className="flex-1" value={text} onChange={(e) => set(e.target.value)} />
            )}
            <ItemTools
              index={index}
              count={items.length}
              label={label}
              onMove={(to) => onChange(move(items, index, to))}
              onRemove={() => onChange(items.filter((_, i) => i !== index))}
            />
          </div>
        );
      })}
      <Button variant="secondary" size="sm" onClick={() => onChange([...items, ""])}>
        <Plus /> Add {noun}
      </Button>
    </div>
  );
}
