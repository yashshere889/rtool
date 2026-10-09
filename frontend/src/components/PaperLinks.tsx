import { Check, Copy, ExternalLink } from "lucide-react";
import { useState } from "react";

import type { PaperCardData } from "@/api/types";

import { Button } from "./ui/button";

/** DOI, arXiv and OpenAlex pages of a paper, opened in a new tab. */
export function ExternalLinks({ links, className }: { links: PaperCardData["links"]; className?: string }) {
  const items = [
    ["DOI", links.doi],
    ["arXiv", links.arxiv],
    ["OpenAlex", links.openalex],
  ].filter((item): item is [string, string] => Boolean(item[1]));
  if (!items.length) return null;
  return (
    <div className={className ?? "flex flex-wrap gap-3 text-sm"}>
      {items.map(([label, href]) => (
        <a key={label} href={href} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-accent hover:underline">
          {label} <ExternalLink className="size-3.5" />
        </a>
      ))}
    </div>
  );
}

export function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      variant="secondary"
      size="sm"
      onClick={async () => {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
    >
      {copied ? <Check /> : <Copy />} {copied ? "Copied" : label}
    </Button>
  );
}
