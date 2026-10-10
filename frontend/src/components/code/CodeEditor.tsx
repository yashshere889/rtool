/** Code and Markdown editors (CodeMirror 6). Loaded on demand: they are the largest part of the app. */
import { markdown } from "@codemirror/lang-markdown";
import { python } from "@codemirror/lang-python";
import { unifiedMergeView } from "@codemirror/merge";
import { Compartment, EditorState, type Extension } from "@codemirror/state";
import { Decoration, EditorView, MatchDecorator, ViewPlugin, type DecorationSet, type ViewUpdate } from "@codemirror/view";
import { oneDark } from "@codemirror/theme-one-dark";
import { basicSetup } from "codemirror";
import { forwardRef, useEffect, useImperativeHandle, useRef, type RefObject } from "react";

import { useDark } from "@/lib/useDark";

// read-only editors are not focusable by default; keyboard users still need to scroll and read them
const focusable = EditorView.contentAttributes.of({ tabindex: "0" });

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
          focusable,
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
  return view;
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

const chipTheme = EditorView.baseTheme({
  ".cm-rt-placeholder": {
    backgroundColor: "color-mix(in srgb, var(--info) 16%, transparent)",
    borderRadius: "4px",
    padding: "0 2px",
  },
  ".cm-rt-citation": {
    backgroundColor: "color-mix(in srgb, var(--ok) 16%, transparent)",
    borderRadius: "4px",
    padding: "0 2px",
  },
  ".cm-rt-unknown": {
    backgroundColor: "color-mix(in srgb, var(--bad) 20%, transparent)",
    textDecoration: "underline wavy var(--bad)",
  },
});

function highlighter(decorator: MatchDecorator) {
  return ViewPlugin.fromClass(
    class {
      decorations: DecorationSet;
      constructor(view: EditorView) {
        this.decorations = decorator.createDeco(view);
      }
      update(update: ViewUpdate) {
        this.decorations = decorator.updateDeco(update, this.decorations);
      }
    },
    { decorations: (plugin) => plugin.decorations },
  );
}

/** {{placeholders}} and [@citations] as chips; the tooltip says what they stand for, unknown ones are marked. */
function chips(placeholders: Map<string, string>, papers: Map<string, string>) {
  const placeholder = new MatchDecorator({
    regexp: /\{\{\s*([A-Za-z_]+(?::[A-Za-z0-9_.-]+)*)\s*\}\}/g,
    decoration: (match) =>
      Decoration.mark({
        class: placeholders.has(match[1]) ? "cm-rt-placeholder" : "cm-rt-unknown",
        attributes: { title: placeholders.get(match[1]) ?? "Unknown placeholder: it would not render" },
      }),
  });
  const citation = new MatchDecorator({
    regexp: /\[\s*@[^\]]+?\]/g,
    decoration: (match) => {
      const keys = [...match[0].matchAll(/@([A-Za-z0-9_:.-]+)/g)].map((k) => k[1]);
      return Decoration.mark({
        class: keys.every((k) => papers.has(k)) ? "cm-rt-citation" : "cm-rt-unknown",
        attributes: { title: keys.map((k) => papers.get(k) ?? `${k}: not a paper in this collection`).join("\n") },
      });
    },
  });
  return [highlighter(placeholder), highlighter(citation), chipTheme];
}

export interface MarkdownEditorHandle {
  insert: (text: string) => void;
}

/** A Markdown editor for a report section, with placeholders and citations shown as chips. */
export const MarkdownEditor = forwardRef<
  MarkdownEditorHandle,
  {
    value: string;
    onChange: (text: string) => void;
    label: string;
    placeholders: Map<string, string>;
    papers: Map<string, string>;
    onFocus?: () => void;
  }
>(function MarkdownEditor({ value, onChange, label, placeholders, papers, onFocus }, ref) {
  const parent = useRef<HTMLDivElement>(null);
  const focused = useRef(onFocus);
  focused.current = onFocus;
  const view = useEditor(
    parent,
    [
      basicSetup,
      markdown(),
      frame,
      EditorView.lineWrapping,
      EditorView.theme({ ".cm-content, .cm-gutter": { minHeight: "8rem" } }),
      EditorView.contentAttributes.of({ "aria-label": label }),
      EditorView.domEventHandlers({ focus: () => void focused.current?.() }),
      chips(placeholders, papers),
    ],
    value,
    onChange,
  );
  useImperativeHandle(ref, () => ({
    insert(text: string) {
      const current = view.current;
      if (!current) return;
      const { from, to } = current.state.selection.main;
      current.dispatch({ changes: { from, to, insert: text }, selection: { anchor: from + text.length } });
      current.focus();
    },
  }));
  return <div ref={parent} className="max-h-[60vh] overflow-auto rounded-xl" />;
});
