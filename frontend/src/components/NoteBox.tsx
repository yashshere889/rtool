import { useState } from "react";

import { useDecide } from "@/api/hooks";
import type { RunDetail } from "@/api/types";
import { stageTitle } from "@/lib/stages";

import { Button } from "./ui/button";
import { Textarea } from "./ui/field";

/** Guidance for an agent, read the next time its stage runs. */
export function NoteBox({ run, stage }: { run: RunDetail; stage: string }) {
  const decide = useDecide(run.project, run.run_id);
  const [text, setText] = useState("");
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        decide.mutate(
          { request: { kind: "note", stage, payload: { text: text.trim() } }, label: `Note for ${stageTitle(stage)}` },
          { onSuccess: () => setText("") },
        );
      }}
    >
      <label htmlFor={`guide-${stage}`} className="text-sm font-medium">
        Guide the {stageTitle(stage).toLowerCase()} agent
      </label>
      <p className="text-xs text-ink-soft">It reads your note the next time this stage runs.</p>
      <Textarea
        id={`guide-${stage}`}
        className="mt-2 min-h-16"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="For example: prefer studies from the last ten years."
      />
      <Button type="submit" variant="secondary" size="sm" className="mt-2" busy={decide.isPending} disabled={!text.trim()}>
        Send note
      </Button>
    </form>
  );
}
