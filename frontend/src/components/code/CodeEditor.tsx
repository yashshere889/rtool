/** A Python editor (CodeMirror 6). Loaded on demand: it is the largest part of the app. */
import { python } from "@codemirror/lang-python";
import { unifiedMergeView } from "@codemirror/merge";
import { Compartment, EditorState, type Extension } from "@codemirror/state";
import { EditorView } from "@codemirror/view";
import { oneDark } from "@codemirror/theme-one-dark";
import { basicSetup } from "codemirror";
import { useEffect, useRef, type RefObject } from "react";

import { useDark } from "@/lib/useDark";

const frame = EditorView.theme({
  "&": { fontSize: "13px", border: "1px solid var(--line)", borderRadius: "0.75rem", overflow: "hidden" },
  ".cm-scroller": { fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", lineHeight: "1.55" },
  "&.cm-focused": { outline: "2px solid color-mix(in srgb, var(--accent) 35%, transparent)" },
});

function useEditor(
  parent: RefObject<HTMLDivElement | null>,
  extensions: Extension[],
  doc: string,
  onChange?: (text: string) => void,
) {
  const view = useRef<EditorView | null>(null);
  const theme = useRef(new Compartment());
  const changed = useRef(onChange);
  changed.current = onChange;
  const dark = useDark();

  useEffect(() => {
    if (!parent.current) return;
    view.current = new EditorView({
      parent: parent.current,
      state: EditorState.create({
        doc,
        extensions: [
          ...extensions,
          theme.current.of(dark ? oneDark : []),
          EditorView.updateListener.of((update) => {
            if (update.docChanged) changed.current?.(update.state.doc.toString());
          }),
        ],
      }),
    });
    return () => view.current?.destroy();
    // the editor is created once; later changes to doc arrive through the effect below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    view.current?.dispatch({ effects: theme.current.reconfigure(dark ? oneDark : []) });
  }, [dark]);

  useEffect(() => {
    const current = view.current;
    if (current && current.state.doc.toString() !== doc) {
      current.dispatch({ changes: { from: 0, to: current.state.doc.length, insert: doc } });
    }
  }, [doc]);
}

export default function CodeEditor({
  value,
  onChange,
  readOnly = false,
  label,
  minHeight = "28rem",
}: {
  value: string;
  onChange?: (text: string) => void;
  readOnly?: boolean;
  label: string;
  minHeight?: string;
}) {
  const parent = useRef<HTMLDivElement>(null);
  useEditor(
    parent,
    [
      basicSetup,
      python(),
      frame,
      EditorView.theme({ ".cm-content, .cm-gutter": { minHeight } }),
      EditorState.readOnly.of(readOnly),
      EditorView.editable.of(!readOnly),
      EditorView.contentAttributes.of({ "aria-label": label }),
    ],
    value,
    onChange,
  );
  return <div ref={parent} className="max-h-[75vh] overflow-auto rounded-xl" />;
}

/** The current file with the changes since ``original`` marked line by line (read-only). */
export function CodeDiff({ original, value, label }: { original: string; value: string; label: string }) {
  const parent = useRef<HTMLDivElement>(null);
  useEditor(
    parent,
    [
      basicSetup,
      python(),
      frame,
      EditorState.readOnly.of(true),
      EditorView.editable.of(false),
      EditorView.contentAttributes.of({ "aria-label": label }),
      unifiedMergeView({ original, mergeControls: false, gutter: true }),
    ],
    value,
  );
  return <div ref={parent} className="max-h-[75vh] overflow-auto rounded-xl" />;
}
