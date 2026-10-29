import { BadgeCheck, CircleAlert, ScanSearch } from "lucide-react";
import { Link } from "react-router";

import type { ClaimView } from "@/api/types";

import { Badge } from "./ui/badge";
import { Tooltip } from "./ui/tooltip";

/** A claim with its supporting quote, and whether the quote was found in the paper's text. */
export function QuoteBlock({ claim, showInPaper }: { claim: ClaimView; showInPaper?: string }) {
  const tags = [claim.method, claim.dataset, claim.metric && claim.value ? `${claim.metric} = ${claim.value}` : claim.metric]
    .filter(Boolean);
  const unquoted = claim.unquoted_numbers ?? [];
  const quoteFound = claim.support >= 0.85; // the agent's threshold: below it the quote does not count as found
  return (
    <figure className="space-y-2">
      <p className="text-sm font-medium">{claim.text}</p>
      <blockquote className="border-l-2 border-accent/40 pl-3 text-sm text-ink-soft italic">“{claim.quote}”</blockquote>
      <figcaption className="flex flex-wrap items-center gap-2">
        {claim.verified ? (
          <Tooltip content="The quote was found in the paper's text, so this claim is grounded in the paper.">
            <span>
              <Badge tone="ok">
                <BadgeCheck className="size-3.5" /> Quote found in the paper ({Math.round(claim.support * 100)}% match)
              </Badge>
            </span>
          </Tooltip>
        ) : quoteFound ? (
          <Tooltip content="The quote is in the paper, but the claim states a number that the paper's text does not contain. Treat this claim with care.">
            <span>
              <Badge tone="bad">
                <CircleAlert className="size-3.5" /> Number not in the paper: {unquoted.join(", ")}
              </Badge>
            </span>
          </Tooltip>
        ) : (
          <Tooltip content="The quote could not be found in the paper. Treat this claim with care.">
            <span>
              <Badge tone="bad">
                <CircleAlert className="size-3.5" /> Quote not found ({Math.round(claim.support * 100)}% match)
              </Badge>
            </span>
          </Tooltip>
        )}
        {claim.verified && unquoted.length > 0 && (
          <Tooltip content="These numbers are elsewhere in the paper, but not in this quote: check them in the paper.">
            <span>
              <Badge tone="warn">Not in this quote: {unquoted.join(", ")}</Badge>
            </span>
          </Tooltip>
        )}
        {tags.map((tag) => (
          <Badge key={tag}>{tag}</Badge>
        ))}
        {showInPaper && (
          <Link to={showInPaper} className="inline-flex items-center gap-1 text-xs font-medium text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent">
            <ScanSearch className="size-3.5" /> Show in paper
          </Link>
        )}
      </figcaption>
    </figure>
  );
}
