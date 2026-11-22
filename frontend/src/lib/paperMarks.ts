import { useCallback, useMemo, useSyncExternalStore } from "react";

export type Choice = "include" | "exclude";
export type Marks = Record<string, Choice>;

// Keep/Drop marks not sent yet. Kept per checkpoint (its nonce) in sessionStorage, so they survive opening a
// paper and coming back, and every page showing that checkpoint sees the same marks.
const listeners = new Set<() => void>();
const memory = new Map<string, string>(); // when sessionStorage is blocked

const keyOf = (project: string, runId: string, nonce: string) => `rtool.marks.${project}/${runId}/${nonce}`;

function read(key: string): string {
  try {
    return sessionStorage.getItem(key) ?? "{}";
  } catch {
    return memory.get(key) ?? "{}";
  }
}

function write(key: string, marks: Marks) {
  const value = JSON.stringify(marks);
  try {
    if (value === "{}") sessionStorage.removeItem(key);
    else sessionStorage.setItem(key, value);
  } catch {
    memory.set(key, value);
  }
  listeners.forEach((notify) => notify());
}

function subscribe(notify: () => void) {
  listeners.add(notify);
  return () => listeners.delete(notify);
}

/** The unsent marks at one checkpoint (``nonce`` null: no open checkpoint, so none). */
export function usePaperMarks(project: string, runId: string, nonce: string | null | undefined) {
  const key = nonce ? keyOf(project, runId, nonce) : null;
  const raw = useSyncExternalStore(subscribe, () => (key ? read(key) : "{}"));
  const marks = useMemo<Marks>(() => {
    try {
      return JSON.parse(raw) as Marks;
    } catch {
      return {};
    }
  }, [raw]);
  const mark = useCallback(
    (id: string, choice: Choice | null) => {
      if (!key) return;
      const next = { ...(JSON.parse(read(key)) as Marks) };
      if (choice === null) delete next[id];
      else next[id] = choice;
      write(key, next);
    },
    [key],
  );
  const clear = useCallback(() => key && write(key, {}), [key]);
  return { marks, mark, clear };
}
