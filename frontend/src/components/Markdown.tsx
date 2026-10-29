import ReactMarkdown, { defaultUrlTransform, type Components } from "react-markdown";
import { Link } from "react-router";
import remarkGfm from "remark-gfm";

import { fileUrl, readerPath } from "@/api/client";
import type { PaperInfo } from "@/api/types";

const CITATION = /\[\s*(@[^\]]+?)\s*\]/g;

/** "[@a; @b]" -> "([Morgan et al., 2019](cite:a); [Novak, 2020](cite:b))", so they render as links to the papers. */
export function linkCitations(text: string, papers: Map<string, PaperInfo>): string {
  return text.replace(CITATION, (_, group: string) => {
    const keys = group.split(";").map((part) => part.trim().replace(/^@/, "")).filter(Boolean);
    return `(${keys.map((k) => `[${papers.get(k)?.label ?? k}](cite:${encodeURIComponent(k)})`).join("; ")})`;
  });
}

/** Report Markdown: tables, figures from the run, and citations that open the paper in the reader. */
export function Markdown({
  text,
  papers,
  project,
  runId,
  stage,
  baseDir,
}: {
  text: string;
  papers: Map<string, PaperInfo>;
  project: string;
  runId: string;
  stage: string; // the reader's "back" link returns to this stage's page
  baseDir: string; // run-relative folder that relative image paths start from
}) {
  const components: Components = {
    a({ href = "", children }) {
      if (href.startsWith("cite:")) {
        const paper = papers.get(decodeURIComponent(href.slice(5)));
        if (!paper) return <span className="text-bad">{children}</span>;
        return (
          <Link to={readerPath(project, runId, stage, paper.paper_id)} title={paper.title} className="rounded bg-ok-soft/70 px-1 text-ok underline decoration-ok/40 underline-offset-2">
            {children}
          </Link>
        );
      }
      return (
        <a href={href} target="_blank" rel="noreferrer">
          {children}
        </a>
      );
    },
    img({ src = "", alt = "" }) {
      const url = typeof src === "string" && !/^[a-z]+:/i.test(src) ? fileUrl(project, runId, `${baseDir}/${src}`) : String(src);
      return <img src={url} alt={alt} loading="lazy" />;
    },
  };
  return (
    <div className="rt-prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={components}
        urlTransform={(url) => (url.startsWith("cite:") ? url : defaultUrlTransform(url))}
      >
        {linkCitations(text, papers)}
      </ReactMarkdown>
    </div>
  );
}
