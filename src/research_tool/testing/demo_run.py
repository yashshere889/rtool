"""Demo runs of the research pipeline, for developing and testing the web UI without a cluster.

    python -m research_tool.testing.demo_run runs/demo     # one run stopped at each checkpoint, plus a finished one

The real agents and engine run end to end; only the outside world is faked:
paper search and downloads (the fictional papers of :mod:`.demo_data`, served
as real PDF and HTML files), the model (scripted answers that follow each
prompt's rules, e.g. quotes copied from the paper text) and Slurm (experiment
jobs really run, locally). Runs carry a ``demo.json`` marker so tools that
advance runs locally (``rtool ui --mode local``) know to use these fakes.
"""

from __future__ import annotations

import argparse
import itertools
import re
import textwrap
import threading
from pathlib import Path
from typing import Any

import httpx

from ..agents.literature.toolkit import LiteratureToolkit
from ..core.actions import ActionKind
from ..core.config import deep_merge, load_config
from ..core.engine import AdvanceResult, Engine, Outcome
from ..core.fs import atomic_write_json
from ..core.run import RunHandle
from ..core.services import Services
from ..hpc.fake_slurm import FakeSlurm
from ..knowledge.vectors import HashingEmbedder
from ..llm.client import LLMClient
from ..llm.mock_backend import MockLLM
from ..llm.types import ChatRequest
from ..service import create_run
from ..tools.documents.convert import ConversionError
from ..tools.http import PoliteHttp
from ..tools.resources.checks import ResourceCheck
from ..tools.resources.fetch import Fetcher
from ..tools.scholarly.base import PaperHit
from . import demo_data as data
from .harness import queue

PROJECT = "sleep-demo"
MARKER = "demo.json"
# A small Barkla-like cluster and GPU budget, so the plan's compute has something real to be checked against.
CLUSTER: dict[str, Any] = {
    "slurm": {
        "partitions": {
            "gpu-l40s": {"gpu_type": "l40s", "gpus_per_node": 2, "gpu_memory_gb": 48, "note": "common"},
            "gpu-h100": {"gpu_type": "h100", "gpus_per_node": 4, "gpu_memory_gb": 80, "note": "common"},
            "gpu-l40s-low": {
                "gpu_type": "l40s",
                "gpus_per_node": 2,
                "gpu_memory_gb": 48,
                "max_time": "1-00:00:00",
                "preemptible": True,
                "note": "low priority",
            },
            "nodes": {"note": "CPU nodes"},
        },
        "experiment_partitions": ["gpu-l40s", "gpu-h100"],
    },
    "budgets": {"run": {"max_gpu_hours": 40}},
}
CHECKPOINTS = ("literature_review", "hypothesis_selection", "experiment_plan", "gpu_runs", "final_report", "done")
RUN_IDS = {
    "literature_review": "1-literature",
    "hypothesis_selection": "2-hypotheses",
    "experiment_plan": "3-plan",
    "gpu_runs": "4-gpu-runs",
    "final_report": "5-final-report",
    "done": "6-done",
}
DECISIONS: tuple[tuple[str, ActionKind, dict[str, Any]], ...] = (
    ("literature_review", ActionKind.APPROVE, {}),
    ("hypothesis_selection", ActionKind.SELECT, {"index": 0}),
    ("experiment_plan", ActionKind.APPROVE, {}),
    ("gpu_runs", ActionKind.APPROVE, {}),
    ("final_report", ActionKind.APPROVE, {}),
)
_PDF_MARKER = re.compile(rb"%rtool-demo-paper:([a-z0-9]+)")


# ---------------------------------------------------------------- documents
def text_pdf(title: str, text: str, *, marker: str = "") -> bytes:
    """A small, valid, multi-page PDF of ``text`` (Markdown headings set in bold)."""
    lines: list[tuple[bool, str]] = []
    for raw in text.splitlines():
        heading = raw.startswith("#")
        body = raw.lstrip("#").strip()
        wrapped = textwrap.wrap(body, 60 if heading else 92) or [""]
        lines += [(heading, line) for line in wrapped]
    per_page = 58
    pages = [lines[i : i + per_page] for i in range(0, len(lines), per_page)] or [[]]

    def escape(value: str) -> str:
        value = value.encode("latin-1", "replace").decode("latin-1")
        return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",  # pages, filled in below
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        f"<< /Title ({escape(title)}) /Producer (rtool demo) >>".encode("latin-1"),
    ]
    kids = []
    for page in pages:
        ops = ["BT", "14 TL", "56 790 Td"]
        for heading, line in page:
            ops += ["/F2 12 Tf" if heading else "/F1 10 Tf", f"({escape(line)}) Tj", "T*"]
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        content_number = len(objects)
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
            b"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents %d 0 R >>" % content_number
        )
        kids.append(len(objects))
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % k for k in kids),
        len(kids),
    )
    out = bytearray(b"%PDF-1.4\n")
    if marker:
        out += f"%rtool-demo-paper:{marker}\n".encode()
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R /Info 5 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def paper_html(paper: data.DemoPaper) -> str:
    """The paper as an arXiv-style HTML page."""
    body = []
    for line in paper.text.splitlines():
        if line.startswith("## "):
            body.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            body.append(f"<h1>{line[2:]}</h1>")
        elif line.strip():
            body.append(f"<p>{line}</p>")
    return "<!DOCTYPE html><html><body><nav>arXiv menu</nav>" + "\n".join(body) + "</body></html>"


def pdf_url(paper: data.DemoPaper) -> str:
    return f"https://papers.demo.example/{paper.key}.pdf"


class DemoDocuments:
    """Serves the open-access copies: arXiv HTML or PDF, publisher PDFs; paywalled papers 404."""

    def __init__(self) -> None:
        self.by_url: dict[str, tuple[str, bytes]] = {}
        for paper in data.PAPERS:
            if paper.access == "none" or not paper.sections:
                continue
            pdf = text_pdf(paper.title, paper.text, marker=paper.key)
            if paper.arxiv_id:
                if paper.access == "html":
                    self.by_url[f"https://arxiv.org/html/{paper.arxiv_id}"] = ("text/html", paper_html(paper).encode())
                else:
                    self.by_url[f"https://arxiv.org/pdf/{paper.arxiv_id}"] = ("application/pdf", pdf)
            else:
                self.by_url[pdf_url(paper)] = ("application/pdf", pdf)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        found = self.by_url.get(str(request.url))
        if found is None:
            return httpx.Response(404)
        content_type, body = found
        return httpx.Response(200, content=body, headers={"content-type": content_type})


def demo_pdf_to_text(path: Path) -> str:
    """What a PDF converter would read from a demo PDF (the paper's Markdown text)."""
    match = _PDF_MARKER.search(Path(path).read_bytes()[:200])
    if match is None or match.group(1).decode() not in data.BY_KEY:
        raise ConversionError(f"{path} is not a demo paper")
    return data.BY_KEY[match.group(1).decode()].text


# ------------------------------------------------------------------- search
def paper_hit(paper: data.DemoPaper) -> PaperHit:
    number = 2_000_000 + data.PAPERS.index(paper)
    oa = [pdf_url(paper)] if paper.access == "pdf" and paper.sections and not paper.arxiv_id else []
    url = f"https://doi.org/{paper.doi}" if paper.doi else f"https://arxiv.org/abs/{paper.arxiv_id}"
    return PaperHit(
        source="openalex",
        source_id=paper.key,
        title=paper.title,
        abstract=paper.abstract,
        year=paper.year,
        authors=list(paper.authors),
        venue=paper.venue,
        doi=paper.doi,
        arxiv_id=paper.arxiv_id,
        s2_id=f"S-{paper.key}",
        openalex_id=f"W{number}",
        url=url,
        oa_urls=oa,
        citation_count=paper.citations,
        reference_ids=[f"W{2_000_000 + data.PAPERS.index(data.BY_KEY[k])}" for k in data.REFERENCES.get(paper.key, ())],
    )


class DemoSearch:
    name = "openalex"

    def search(self, query: str, limit: int, *, year_from: int | None = None) -> list[PaperHit]:
        return [paper_hit(p) for p in data.PAPERS if not p.found_by_snowballing][:limit]


class DemoSemanticScholar(DemoSearch):
    name = "semantic_scholar"

    def search(self, query: str, limit: int, *, year_from: int | None = None) -> list[PaperHit]:
        return []

    @staticmethod
    def reference_for(paper: Any) -> str | None:
        return paper.s2_id

    def references(self, ref: str, limit: int) -> list[PaperHit]:
        key = ref.removeprefix("S-")
        return [paper_hit(data.BY_KEY[k]) for k in data.REFERENCES.get(key, ())][:limit]

    def citations(self, ref: str, limit: int) -> list[PaperHit]:
        return []


# -------------------------------------------------------------------- model
def _schema(request: ChatRequest) -> str:
    return str((request.json_schema or {}).get("title", ""))


def _last(request: ChatRequest) -> str:
    return request.messages[-1].content if request.messages else ""


class DemoModel:
    """Scripted answers that obey each prompt's rules, so every check in the pipeline passes for real."""

    def __init__(self) -> None:
        self._drafts = itertools.count()
        self._lock = threading.Lock()

    # ---- literature
    def relevance(self, request: ChatRequest) -> dict[str, Any]:
        match = re.search(r"^Paper: (.+?) \(", _last(request), re.M)
        paper = data.BY_TITLE.get(match.group(1)) if match else None
        if paper is None:
            return {"score": 3, "reason": "Same broad field only."}
        title = paper.title.lower()  # the demo plan's sub-topics, in order (see demo_llm)
        covers = [n for n, word in enumerate(("stimulation", "spindle", "headband"), 1) if word in title]
        return {"score": paper.relevance, "reason": paper.reason, "subtopics": covers}

    def extraction(self, request: ChatRequest) -> dict[str, Any]:
        match = re.search(r"^Paper title: (.+)$", _last(request), re.M)
        paper = data.BY_TITLE.get(match.group(1).strip()) if match else None
        if paper is None:
            return {"problem": "", "approach": "", "key_results": [], "limitations": []}
        return paper.extraction()

    @staticmethod
    def digest(request: ChatRequest) -> dict[str, Any]:
        match = re.search(r"^Paper title: (.+)$", _last(request), re.M)
        paper = data.BY_TITLE.get(match.group(1).strip()) if match else None
        if paper is None:
            return {"summary": "not stated", "design": "not stated", "sample": "not stated", "procedure": "not stated"}
        return paper.digest()

    @staticmethod
    def topic_label(request: ChatRequest) -> dict[str, Any]:
        titles = " ".join(re.findall(r"^- (.+?) \(\d{4}", _last(request), re.M)).lower()
        if "older" in titles or "ageing" in titles or "cognitive impairment" in titles:
            return {"label": "Stimulation across ages", "summary": "How age and clinical status change the effect."}
        if "spindle" in titles or "coupling" in titles:
            return {"label": "Spindles and coupling", "summary": "Spindles, their timing and memory retention."}
        if "detection" in titles or "model" in titles:
            return {"label": "Methods and models", "summary": "Detection algorithms and computational models."}
        return {"label": "Closed-loop stimulation", "summary": "Sound timed to slow oscillations and its effects."}

    @staticmethod
    def open_problems(request: ChatRequest) -> dict[str, Any]:
        lines = re.findall(r"^\[([^\]]+)\] (.+)$", _last(request), re.M)
        ids = list(dict.fromkeys(pid for pid, _ in lines))

        def mentioning(*words: str) -> list[str]:
            hits = [pid for pid, text in lines if any(w in text.lower() for w in words)]
            return list(dict.fromkeys(hits)) or ids[:1]

        return {
            "problems": [
                {
                    "statement": "Whether stimulation helps older adults and clinical groups as it helps young adults",
                    "paper_ids": mentioning("older", "young", "clinical", "small sample"),
                    "why_open": "Most studies tested young, healthy students, and the few in older groups were small",
                },
                {
                    "statement": "Whether the memory benefit lasts beyond a single night of stimulation",
                    "paper_ids": mentioning("night"),
                    "why_open": "Studies tested memory only the morning after one or a few nights",
                },
                {
                    "statement": "Whether slow oscillation-spindle coupling causes the memory benefit",
                    "paper_ids": mentioning("coupling", "correlational", "causes"),
                    "why_open": "The evidence for coupling is correlational; it has not been manipulated directly",
                },
            ]
        }

    @staticmethod
    def overview(request: ChatRequest) -> dict[str, Any]:
        keys = re.findall(r"^\[([a-z0-9]+)\]", _last(request), re.M) + ["", "", "", ""]
        return {
            "text": (
                f"Sounds timed to the up-state of slow oscillations reliably enlarge these oscillations and improve "
                f"overnight recall in young adults [{keys[0]}]. A direct comparison found the benefit only for "
                f"phase-locked stimulation, while untimed sounds fragmented sleep [{keys[1]}].\n\n"
                f"Mechanistic work points to the coupling between slow oscillations and spindles [{keys[2]}], and "
                f"the effect appears weaker in older adults [{keys[3]}].\n\n"
                "Open questions concern lasting benefits after several nights, older and clinical groups, and "
                "whether coupling causes the memory effect."
            ).replace(" []", "")
        }

    # ---- hypotheses
    def drafts(self, request: ChatRequest) -> dict[str, Any]:
        prompt = _last(request)
        evidence_block = prompt.split("Other papers you may cite:", 1)[0]
        evidence = re.findall(r"^\[([^\]]+)\] (.+)$", evidence_block, re.M)
        with self._lock:
            template = dict(data.HYPOTHESES[next(self._drafts) % len(data.HYPOTHESES)])
        template.pop("scores")
        template.pop("comment")
        template["evidence"] = [{"paper_id": pid, "quote": quote} for pid, quote in evidence[:2]]
        return {"hypotheses": [template]}

    @staticmethod
    def novelty(request: ChatRequest) -> dict[str, Any]:
        neighbours = re.findall(r"^\[([^\]]+)\] (.+?) \(", _last(request), re.M)
        closest = neighbours[0][0] if neighbours else ""
        title = neighbours[0][1] if neighbours else "the closest paper"
        return {
            "novel": True,
            "closest_paper_id": closest,
            "overlap": f"Related to '{title}', which did not test this comparison in this setting.",
            "score": 7,
        }

    @staticmethod
    def rubric(request: ChatRequest) -> dict[str, Any]:
        match = re.search(r"^Hypothesis: (.+)$", _last(request), re.M)
        name = match.group(1).strip() if match else ""
        for template in data.HYPOTHESES:
            if template["title"] == name:
                return {**template["scores"], "comment": template["comment"]}
        return {"novelty": 3, "feasibility": 3, "testability": 3, "significance": 3, "clarity": 3, "comment": "ok"}

    # ---- report
    @staticmethod
    def section(request: ChatRequest) -> str:
        prompt = _last(request)
        title = re.search(r'"([^"]+)" section', prompt)
        name = title.group(1) if title else ""
        keys = [k for k in re.findall(r"@([A-Za-z0-9_:.\-]+)", prompt) if k not in ("key", "a", "b")]
        keys = list(dict.fromkeys(keys))
        if name == "Results":
            return _results_text(prompt)
        text = data.SECTION_TEXT.get(name, "This section summarises the study.")

        def cite(match: re.Match[str]) -> str:  # cite the intended paper if its key was offered, else nothing
            key = next((k for k in keys if k.startswith(match.group(1))), None)
            return f" [@{key}]" if key else ""

        return re.sub(r"\{cite:([a-z0-9]+)\}", cite, text)


def _results_text(prompt: str) -> str:
    available = set(re.findall(r"^- \{\{([a-z_]+(?::[a-z0-9_]+)*)\}\}", prompt, re.M))
    paragraphs = []
    for metric, label in (("recall_gain", "recall gain"), ("so_amplitude", "slow-oscillation amplitude")):
        sentences = []
        for condition, name in (("phase_locked", "Phase-locked stimulation"), ("fixed_interval", "Fixed-interval")):
            key = f"{metric}:{condition}"
            if f"mean:{key}" not in available:
                continue
            sentence = f"{name} reached a mean {label} of {{{{mean:{key}}}}}"
            if f"ci:{key}" in available:
                sentence += f" ({{{{ci:{key}}}}})"
            if f"diff:{key}" in available and f"p:{key}" in available:
                sentence += f"; the difference to sham was {{{{diff:{key}}}}} ({{{{p:{key}}}}})"
            sentences.append(sentence + ".")
        if sentences:
            paragraphs.append(" ".join(sentences))
    return "\n\n".join(paragraphs) or "The experiment results are summarised in the tables."


DEMO_SCOPE = {
    "interpretation": (
        "Whether sounds played in time with the slow oscillations of deep sleep strengthen memory consolidation, "
        "and how the timing of the sounds matters."
    ),
    "concepts": ["auditory stimulation during sleep", "slow oscillations", "memory consolidation"],
    "out_of_scope": ["sleep apnoea treatment", "music and sleep onset"],
    "other_readings": [],
}


_AUDIT_QUESTION = re.compile(r'^\d+\. aspect "(\w+)"(?:, subject "([^"]*)")?', re.M)


def _code_audit(request: ChatRequest) -> dict[str, Any]:
    """Every code-audit question answered ok: the demo's experiment does what its plan says."""
    return {"answers": [{"aspect": a, "subject": s, "ok": True} for a, s in _AUDIT_QUESTION.findall(_last(request))]}


def demo_llm() -> MockLLM:
    model = DemoModel()
    llm = MockLLM()
    structured = {
        "TopicScope": DEMO_SCOPE,
        "SearchPlan": {
            "queries": data.SEARCH_QUERIES,
            "must_cover": ["closed-loop stimulation", "spindles", "stimulation at home"],
        },
        "RelevanceJudgment": model.relevance,
        "PaperExtraction": model.extraction,
        "PaperDigest": model.digest,
        "TopicLabel": model.topic_label,
        "OpenProblems": model.open_problems,
        "Overview": model.overview,
        "HypothesisDrafts": model.drafts,
        "NoveltyJudgment": model.novelty,
        "RubricScores": model.rubric,
        "PlanCore": data.PLAN_CORE,
        "ResourcePlan": {"resources": data.RESOURCES},
        "ComputeEstimate": data.COMPUTE,
        "TitleAbstract": {
            "title": "Phase-locked versus fixed-interval auditory stimulation for sleep-dependent memory",
            "abstract": (
                "Sounds timed to slow oscillations may strengthen memory consolidation during sleep. We compared "
                "phase-locked clicks, fixed-interval clicks and sham in a simulation on public sleep recordings. "
                "Across {{runs_ok}} of {{runs_total}} planned runs, phase-locked stimulation gave the largest "
                "predicted recall gain. Timing appears to matter for stimulation aimed at memory."
            ),
        },
        "HypothesisVerdict": {
            "checks": ["Phase-locked stimulation gave a larger predicted recall gain than sham, as the criterion asks"],
            "outcome": "supported",
        },
        "Contradictions": {"items": []},
        "CodeAuditDraft": _code_audit,
        "ReviewDraft": {
            "scores": {"soundness": 4, "evidence": 4, "clarity": 4, "reproducibility": 5, "limitations": 4},
            "summary": "A careful simulation study; the conclusions follow from the results it reports.",
            "issues": [
                {
                    "section": "results",
                    "problem": "The text reports significance without naming the test, so it cannot be checked",
                    "suggestion": "Say that paired t-tests compared each condition with sham",
                    "target": "report",
                    "severity": "major",
                }
            ],
        },
    }
    for title, reply in structured.items():
        llm.add(lambda r, t=title: _schema(r) == t, reply)
    code = f"Here is the file.\n```python\n{data.EXPERIMENT}```\n"
    llm.add("Write the complete Python file `experiment.py`", code)
    llm.add("fails. Fix it", code)
    llm.add("Revise the experiment code below", code)
    llm.add(
        "You are editing the",
        lambda r: model.section(r) + "\n\nEach condition was compared with sham using a paired t-test.",
    )
    llm.add(lambda r: _last(r).startswith("Rewrite this"), model.section)
    llm.add("section of a research report", model.section)
    return llm


# ---------------------------------------------------------------- resources
class DemoChecker:
    def check(self, resource: Any) -> ResourceCheck:
        return ResourceCheck(True, license="cc-by-4.0", size_gb=1.4)


def demo_snapshot(repo_id: str, repo_type: str, local_dir: str, token: str | None) -> str:
    target = Path(local_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "README.md").write_text(f"Demo copy of {repo_id} (not real data).\n")
    (target / "recordings.csv").write_text("night,n3_minutes\n1,84\n2,91\n")
    return local_dir


# ----------------------------------------------------------------- services
def demo_services(run: RunHandle) -> Services:
    """Fakes for everything outside rtool; the agents and engine are the real ones."""
    documents = DemoDocuments()

    def toolkit(config: Any, cache_path: Path) -> LiteratureToolkit:
        http = PoliteHttp(
            client=httpx.Client(transport=httpx.MockTransport(documents)), sleep=lambda _s: None, default_interval_s=0.0
        )
        return LiteratureToolkit(
            sources={"openalex": DemoSearch(), "semantic_scholar": DemoSemanticScholar()},
            unpaywall=None,
            http=http,
            pdf_converter=demo_pdf_to_text,
            embedder=HashingEmbedder(),
        )

    return Services(
        llm=LLMClient(demo_llm(), run.config.llm),
        slurm=FakeSlurm(run.jobs_dir / "fake_slurm.json", run_commands=True),
        tools={
            "literature_toolkit": toolkit,
            "resource_checker": DemoChecker(),
            "resource_fetcher": lambda root: Fetcher(root, snapshot=demo_snapshot),
        },
    )


def is_demo(run: RunHandle) -> bool:
    return (run.root / MARKER).is_file()


def advance(run: RunHandle, services: Services, *, max_episodes: int = 30) -> AdvanceResult:
    """Episodes until the run needs a person (fake Slurm jobs are run and finished on the way)."""
    result = AdvanceResult(Outcome.NEEDS_HUMAN)
    for _ in range(max_episodes):
        result = Engine(run, services).advance()
        if result.outcome is Outcome.WAITING_JOB and isinstance(services.slurm, FakeSlurm):
            services.slurm.finish_all()
            continue
        if result.outcome in (Outcome.CONTINUE, Outcome.NEEDS_LLM):
            continue
        return result
    return result


# ------------------------------------------------------------------- driver
def create_demo_run(
    runs_root: Path,
    *,
    project: str = PROJECT,
    run_id: str | None = None,
    topic: str = "",
    questions: list[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> RunHandle:
    """A new research run that uses the demo fakes (any topic; the papers are always the demo papers)."""
    settings = deep_merge(deep_merge(CLUSTER, dict(overrides or {})), {"paths": {"runs_root": str(runs_root)}})
    config = load_config(overrides=settings, env={})
    run = create_run(
        config,
        project,
        topic or data.TOPIC,
        questions=data.QUESTIONS if questions is None and not topic else (questions or []),
        run_id=run_id,
        created_by="demo",
    )
    atomic_write_json(run.root / MARKER, {"scenario": "closed-loop auditory stimulation (fictional papers)"})
    return run


def build_demo_run(
    runs_root: Path, stop_at: str = "done", *, project: str = PROJECT, run_id: str | None = None
) -> RunHandle:
    """A demo run advanced through the real pipeline until checkpoint ``stop_at`` is open (or to the end)."""
    if stop_at not in CHECKPOINTS:
        raise ValueError(f"stop_at must be one of {CHECKPOINTS}")
    run = create_demo_run(runs_root, project=project, run_id=run_id or RUN_IDS[stop_at])
    services = demo_services(run)
    for gate, kind, payload in DECISIONS:
        result = advance(run, services)
        opened = [g["gate"] for g in result.open_gates]
        if gate not in opened:
            raise RuntimeError(f"expected checkpoint {gate!r}, got {result.outcome.value}: {result.message} {opened}")
        if stop_at == gate:
            return run
        queue(run, kind, payload=payload)
    result = advance(run, services)
    if result.outcome is not Outcome.DONE:
        raise RuntimeError(f"demo run did not finish: {result.outcome.value}: {result.message}")
    return run


def build_demo_runs(runs_root: Path) -> list[RunHandle]:
    """One run stopped at each checkpoint, plus a finished one."""
    return [build_demo_run(runs_root, stop_at) for stop_at in CHECKPOINTS]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs_root", type=Path, help="Where to create the demo runs (e.g. runs/demo).")
    parser.add_argument("--stop-at", choices=CHECKPOINTS, help="Build only the run stopped at this checkpoint.")
    args = parser.parse_args(argv)
    runs = [build_demo_run(args.runs_root, args.stop_at)] if args.stop_at else build_demo_runs(args.runs_root)
    for run in runs:
        print(f"{run.spec.project}/{run.run_id}: {run.root}")


if __name__ == "__main__":
    main()
