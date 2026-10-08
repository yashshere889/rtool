"""Helpers shared by several agents."""

from __future__ import annotations

from ..core.config import AppConfig
from ..core.context import AgentContext
from ..knowledge.store import KnowledgeStore
from ..schemas.literature import Corpus

SYSTEM = "You are a careful research assistant. Follow the instructions exactly and never invent facts or citations."


def compute_profile(config: AppConfig) -> str:
    """Plain-language description of the GPUs the experiments may use (for planning prompts)."""
    slurm = config.slurm
    names = slurm.experiment_partitions or [p for p, spec in slurm.partitions.items() if spec.gpus_per_node]
    lines = []
    for name in names:
        spec = slurm.partitions.get(name)
        if spec is None:
            lines.append(f"- {name}")
            continue
        memory = f" {spec.gpu_memory_gb:g} GB" if spec.gpu_memory_gb else ""
        gpus = f"{spec.gpus_per_node}x {spec.gpu_type or 'GPU'}{memory} per node, " if spec.gpus_per_node else ""
        lines.append(f"- {name}: {gpus}max {spec.max_time} per job")
    budget = config.budgets.run.max_gpu_hours
    if budget is not None:
        lines.append(f"- total GPU budget for this project: {budget:g} GPU-hours")
    return "\n".join(lines) or "- (no GPU partitions configured: assume a single modern GPU, up to 3 days)"


def open_corpus_kb(ctx: AgentContext, corpus: Corpus) -> KnowledgeStore | None:
    """Read-only knowledge-base snapshot that came with the corpus (None if it is missing)."""
    path = ctx.absolute(corpus.kb_path)
    return KnowledgeStore(path, readonly=True) if path.exists() else None
