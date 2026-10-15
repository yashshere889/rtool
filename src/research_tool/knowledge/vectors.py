"""Small vector toolkit: embeddings, similarity search and clustering.

Uses numpy when it is installed (Barkla) and plain Python otherwise; corpora
are ~100 papers, so either is fast enough.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "a", "an", "the", "of", "on", "in", "for", "and", "or", "to", "with", "by", "via", "from", "as", "at",
        "is", "are", "was", "were", "be", "been", "this", "that", "these", "those", "we", "our", "their", "its",
        "it", "which", "using", "use", "used", "based", "new", "show", "shows", "results", "paper", "study",
    ]
)  # fmt: skip

Vector = list[float]


def unit(vector: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else list(vector)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def similarity_matrix(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    """Cosine similarities of unit vectors."""
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return [[dot(a, b) for b in vectors] for a in vectors]
    matrix = np.asarray(vectors, dtype=float)
    return (matrix @ matrix.T).tolist()


def top_k(query: Sequence[float], vectors: Sequence[Sequence[float]], k: int) -> list[tuple[int, float]]:
    scored = [(i, dot(query, v)) for i, v in enumerate(vectors)]
    return sorted(scored, key=lambda item: item[1], reverse=True)[:k]


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


class HashingEmbedder:
    """Dependency-free lexical embedding (feature-hashed words and word pairs).

    Not semantic like a neural model, but deterministic and good enough to group
    papers by shared vocabulary; it is the fallback when no model is available.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[str]:
        words = [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        out = []
        for text in texts:
            vector = [0.0] * self.dim
            for feature in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
                vector[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            out.append(unit(vector))
        return out


class FastEmbedEmbedder:
    """Small neural embedding model via fastembed (ONNX, CPU). Barkla only; downloads the model once."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding  # type: ignore[import-not-found]

        self._model = TextEmbedding(model_name=model, cache_dir=cache_dir)
        self.name = f"fastembed:{model}"

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [unit([float(x) for x in vector]) for vector in self._model.embed(list(texts))]


def make_embedder(
    kind: str = "hashing", model: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None
) -> Embedder:
    if kind == "fastembed":
        return FastEmbedEmbedder(model, cache_dir)
    return HashingEmbedder()


def agglomerative(vectors: Sequence[Sequence[float]], n_clusters: int) -> list[list[int]]:
    """Average-linkage clustering on cosine similarity; returns groups of indices (largest first)."""
    n = len(vectors)
    if n == 0:
        return []
    n_clusters = max(1, min(n_clusters, n))
    sim = similarity_matrix(vectors)
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    link: dict[tuple[int, int], float] = {(i, j): sim[i][j] for i in range(n) for j in range(i + 1, n)}
    next_id = n
    while len(clusters) > n_clusters:
        (a, b), _ = max(link.items(), key=lambda item: item[1])
        size_a, size_b = len(clusters[a]), len(clusters[b])
        merged = clusters.pop(a) + clusters.pop(b)
        new_links = {}
        for c in clusters:
            s_a = link[(min(a, c), max(a, c))]
            s_b = link[(min(b, c), max(b, c))]
            new_links[(c, next_id)] = (size_a * s_a + size_b * s_b) / (size_a + size_b)  # Lance-Williams
        link = {key: value for key, value in link.items() if a not in key and b not in key}
        link.update(new_links)
        clusters[next_id] = merged
        next_id += 1
    return sorted((sorted(members) for members in clusters.values()), key=len, reverse=True)


def absorb_small(groups: list[list[int]], vectors: Sequence[Sequence[float]], min_size: int = 3) -> list[list[int]]:
    """Move the members of groups smaller than ``min_size`` into the most similar larger group.

    Average linkage leaves outliers as groups of one or two; on the first real run (2026-10-12) that gave one
    90-paper topic and six tiny ones, each then reported as a "thin topic". An outlier is a paper, not a topic.
    """
    large = [list(g) for g in groups if len(g) >= min_size]
    if not large:
        return groups
    for group in (g for g in groups if len(g) < min_size):
        for member in group:
            best = max(
                range(len(large)),
                key=lambda i: sum(dot(vectors[member], vectors[j]) for j in large[i]) / len(large[i]),
            )
            large[best].append(member)
    return sorted((sorted(g) for g in large), key=len, reverse=True)


def suggested_clusters(n_items: int) -> int:
    return max(1, min(12, round(math.sqrt(n_items / 2))))
