import { forwardRef, useEffect, useState, type InputHTMLAttributes, type ReactNode, type TextareaHTMLAttributes } from "react";

import { cn } from "@/lib/cn";

const control =
  "w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-soft/70 " +
  "focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/25 disabled:opacity-60";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function Input(
  { className, ...props },
  ref,
) {
  return <input ref={ref} className={cn(control, "h-9", className)} {...props} />;
});

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(function Textarea(
  { className, ...props },
  ref,
) {
  return <textarea ref={ref} className={cn(control, "min-h-24 leading-relaxed", className)} {...props} />;
});

export function Field({
  label,
  hint,
  htmlFor,
  children,
  className,
}: {
  label: ReactNode;
  hint?: ReactNode;
  htmlFor?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("space-y-1.5", className)}>
      <label htmlFor={htmlFor} className="block text-sm font-medium">
        {label}
      </label>
      {children}
      {hint && <p className="text-xs text-ink-soft">{hint}</p>}
    </div>
  );
}

/** A number box that keeps what is typed (e.g. empty while retyping) and reports only valid numbers. */
export function NumberInput({
  value,
  onValue,
  ...props
}: Omit<InputHTMLAttributes<HTMLInputElement>, "value" | "onChange" | "type"> & {
  value: number | null;
  onValue: (value: number) => void;
}) {
  const [text, setText] = useState(value === null ? "" : String(value));
  useEffect(() => {
    if (Number(text) !== value) setText(value === null ? "" : String(value));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only follow changes made elsewhere
  }, [value]);
  return (
    <Input
      {...props}
      type="number"
      value={text}
      onChange={(e) => {
        setText(e.target.value);
        if (e.target.value.trim() !== "" && Number.isFinite(Number(e.target.value))) onValue(Number(e.target.value));
      }}
    />
  );
}
