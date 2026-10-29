import { cn } from "@/lib/cn";

/** A unified diff, line by line: removed text in red, added in green. */
export function DiffText({ text }: { text: string }) {
  const lines = text.split("\n");
  return (
    <pre className="max-h-[70vh] overflow-auto rounded-xl border border-line bg-surface py-2 font-mono text-xs leading-relaxed">
      {lines.map((line, i) => {
        const header = line.startsWith("+++") || line.startsWith("---");
        const kind = header ? "header" : line.startsWith("@@") ? "hunk" : line.startsWith("+") ? "add" : line.startsWith("-") ? "del" : "same";
        return (
          <div
            key={i}
            className={cn(
              "px-3 whitespace-pre-wrap",
              kind === "add" && "bg-ok-soft/70 text-ok",
              kind === "del" && "bg-bad-soft/70 text-bad",
              (kind === "hunk" || kind === "header") && "text-ink-soft",
            )}
          >
            <span className="sr-only">{kind === "add" ? "added: " : kind === "del" ? "removed: " : ""}</span>
            {line || " "}
          </div>
        );
      })}
    </pre>
  );
}
