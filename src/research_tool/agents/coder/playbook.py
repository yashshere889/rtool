"""Failure playbook: recognise common experiment failures from the log, before asking the LLM.

Deterministic fixes come first because they are cheap and reliable:
a missing module becomes a requirement, running out of memory halves
``BATCH_SIZE``, a time limit grows the job's limit. Everything else is shown
to the LLM together with the end of the log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# import name -> pip package, where they differ
IMPORT_TO_PIP = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python-headless",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "skimage": "scikit-image",
    "Bio": "biopython",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "mne_connectivity": "mne-connectivity",
    "torch_geometric": "torch-geometric",
    "sentence_transformers": "sentence-transformers",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
}
_REQUIREMENT = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9_,.-]+\])?((==|>=|<=|~=|!=|<|>)[A-Za-z0-9.*+!-]+)?$"
)
_MISSING = re.compile(r"No module named '([A-Za-z0-9_.]+)'")
_BATCH = re.compile(r"^(BATCH_SIZE\s*=\s*)(\d+)", re.M)
_REQUIREMENTS_LINE = re.compile(r"^#[ \t]*requirements?[ \t]*:[ \t]*(.*)$", re.M | re.I)  # never past the line end
_EXCEPTION = re.compile(r"^(\w+(?:\.\w+)*(?:Error|Exception|Exit|Interrupt)\b.*)$", re.M)


@dataclass(frozen=True)
class Diagnosis:
    signature: str  # stable name: missing_module, cuda_oom, cpu_oom, time_limit, contract, syntax_error, ...
    summary: str  # one line, shown to the LLM and the human
    requirement: str | None = None  # missing_module: the pip package to add
    memory: bool = False  # out of memory: try a smaller batch first

    @property
    def deterministic(self) -> bool:
        return self.signature in ("missing_module", "cuda_oom", "cpu_oom", "time_limit")


def diagnose(log: str, *, exit_code: int | None = None, timed_out: bool = False, job_state: str = "") -> Diagnosis:
    """Name the failure in a run's log (and exit/job state)."""
    if timed_out or job_state == "TIMEOUT" or "DUE TO TIME LIMIT" in log:
        return Diagnosis("time_limit", "the run hit its time limit")
    missing = _MISSING.findall(log)
    if missing:
        module = missing[-1]
        top = module.split(".", 1)[0]
        if top == "experiment":
            return Diagnosis("contract", "experiment.py could not be imported")
        return Diagnosis(
            "missing_module", f"module {module!r} is not installed", requirement=IMPORT_TO_PIP.get(top, top)
        )
    if re.search(r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|CUDA error: out of memory", log):
        return Diagnosis("cuda_oom", "the GPU ran out of memory", memory=True)
    if job_state == "OUT_OF_MEMORY" or exit_code in (137, -9) or re.search(r"\bMemoryError\b|oom[-_]kill", log):
        return Diagnosis("cpu_oom", "the run ran out of CPU memory", memory=True)
    contract = re.findall(r"rtool-contract: (.+)", log)
    if contract:
        return Diagnosis("contract", contract[-1].strip()[:300])
    if re.search(r"\b(SyntaxError|IndentationError|TabError)\b", log):
        return Diagnosis("syntax_error", _last_exception(log) or "syntax error")
    if "NotImplementedError" in log:
        return Diagnosis("not_implemented", "experiment.run is not implemented yet")
    if re.search(r"FileNotFoundError|No such file or directory", log):
        return Diagnosis("file_not_found", _last_exception(log) or "a file was not found")
    if "Traceback (most recent call last)" in log:
        return Diagnosis("traceback", _last_exception(log) or "the run raised an exception")
    return Diagnosis("failed", f"the run failed (exit code {exit_code})")


def _last_exception(log: str) -> str:
    found = _EXCEPTION.findall(log)
    return found[-1].strip()[:300] if found else ""


# ------------------------------------------------------------- requirements
def valid_requirement(text: str) -> bool:
    """A plain package spec, never a pip option or URL (the text comes from the LLM)."""
    return bool(_REQUIREMENT.match(text.strip()))


def declared_requirements(source: str) -> list[str]:
    """Packages listed in a ``# requirements: a, b`` line of experiment.py."""
    found: list[str] = []
    for line in _REQUIREMENTS_LINE.findall(source):
        for item in re.split(r"[,\s]+", line.strip()):
            if item and valid_requirement(item) and item not in found:
                found.append(item)
    return found


def merge_requirements(*groups: list[str], provided: list[str] = ()) -> list[str]:
    """Unique requirements; packages the image already provides are dropped (no second torch)."""
    have = {_name(p) for p in provided}
    merged: list[str] = []
    for group in groups:
        for item in group:
            if _name(item) not in have and item not in merged:
                merged.append(item)
    return merged


def _name(requirement: str) -> str:
    return re.split(r"[\[<>=!~]", requirement, maxsplit=1)[0].strip().lower().replace("_", "-")


# ------------------------------------------------------------------- memory
def halve_batch_size(source: str) -> str | None:
    """experiment.py with ``BATCH_SIZE = n`` halved, or None if there is no such constant (or it is 1)."""
    match = _BATCH.search(source)
    if not match or int(match.group(2)) <= 1:
        return None
    return _BATCH.sub(lambda m: f"{m.group(1)}{max(1, int(m.group(2)) // 2)}", source, count=1)
