import { Search } from "lucide-react";
import { Fragment, useState } from "react";
import { Link } from "react-router";

import { readerPath } from "@/api/client";
import { useLiteratureSearch } from "@/api/hooks";
import type { RunDetail } from "@/api/types";
import { Input } from "@/components/ui/field";
import { ErrorNote, Loading } from "@/components/ui/states";

function Marked({ text, query }: { text: string; query: string }) {
  const terms = query.toLowerCase().match(/[a-z0-9]{3,}/g) ?? [];
  if (!terms.length) return <>{text}</>;
  const pattern = new RegExp(`(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
  return (
    <>
      {text.split(pattern).map((part, i) =>
        i % 2 ? (
          <mark key={i} className="rounded bg-warn-soft px-0.5 text-ink">
            {part}
          </mark>
        ) : (
          <Fragment key={i}>{part}</Fragment>
        ),
      )}
    </>
  );
}

export function SearchTab({ run, stage }: { run: RunDetail; stage: string }) {
  const [draft, setDraft] = useState("");
  const [query, setQuery] = useState("");
  const results = useLiteratureSearch(run.project, run.run_id, query);
  return (
    <div className="space-y-4">
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          setQuery(draft.trim());
        }}
        role="search"
      >
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute top-2.5 left-3 size-4 text-ink-soft" />
          <Input
            aria-label="Search the full text of every paper"
            className="pl-9"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Search the full text of every paper, e.g. “older adults detection threshold”"
          />
        </div>
      </form>
      {query.length >= 2 && results.isPending && <Loading label="Searching" />}
      {results.isError && <ErrorNote error={results.error} />}
      {results.data && (
        <ol className="space-y-3">
          {results.data.length === 0 && <p className="text-sm text-ink-soft">No passage matches “{query}”.</p>}
          {results.data.map((hit, i) => (
            <li key={`${hit.paper_id}-${i}`} className="rounded-lg border border-line bg-surface p-3">
              <Link to={readerPath(run.project, run.run_id, stage, hit.paper_id)} className="font-medium text-accent hover:underline">
                {hit.title}
              </Link>
              <p className="text-xs text-ink-soft">{hit.section}</p>
              <p className="mt-1 text-sm leading-relaxed">
                <Marked text={hit.snippet} query={query} />
              </p>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
