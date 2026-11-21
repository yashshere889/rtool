"""Helpers shared by several agents."""

from __future__ import annotations

from pathlib import Path

from ..core.config import AppConfig
from ..core.context import AgentContext
from ..core.errors import StageFailed
from ..knowledge.store import KnowledgeStore
from ..packs import Pack, pack_refusal
from ..schemas.literature import Corpus
from ..tools.resources.fetch import local_models, tree_size

SYSTEM = "You are a careful research assistant. Follow the instructions exactly and never invent facts or citations."


def refuse(pack: Pack, what: str, *texts: str) -> None:
    """Stop the stage if rtool refuses these texts: a person must look; it is not a plan issue to approve past."""
    reason = pack_refusal(pack, *texts)
    if reason:
        raise StageFailed(f"rtool refuses to {what}: {reason}", needs_human=True)


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
    models = cached_models(config)
    if models:
        lines.append(f"- models already on the cluster (no download, no token): {', '.join(models)}")
    budget = config.budgets.run.max_gpu_hours
    if budget is not None:
        lines.append(f"- total GPU budget for this project: {budget:g} GPU-hours")
    return "\n".join(lines) or "- (no GPU partitions configured: assume a single modern GPU, up to 3 days)"


def cached_models(config: AppConfig) -> list[str]:
    """The Hub models the cluster already holds, with sizes: a plan that names one downloads nothing."""
    return [
        f"{hub_id} ({tree_size(Path(path))[0] / 1e9:.0f} GB)"
        for hub_id, path in local_models(config.llm.server.hf_home).items()
    ]


def open_corpus_kb(ctx: AgentContext, corpus: Corpus) -> KnowledgeStore | None:
    """Read-only knowledge-base snapshot that came with the corpus (None if it is missing)."""
    path = ctx.absolute(corpus.kb_path)
    return KnowledgeStore(path, readonly=True) if path.exists() else None
