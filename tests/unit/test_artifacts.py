from __future__ import annotations

from pathlib import Path

import pytest

from research_tool.core.artifacts import ArtifactStore, Producer, StagedOutputs
from research_tool.core.errors import ArtifactError
from research_tool.testing.toy_agents import ToyNumbers, ToyStats

P = Producer(stage="s", agent="a")


def test_put_get_and_versioning(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    assert store.put("toy_numbers", ToyNumbers(values=[1.0]), producer=P) == 1
    assert store.put("toy_numbers", ToyNumbers(values=[2.0]), producer=P) == 2
    assert store.get("toy_numbers").values == [2.0]
    assert store.get("toy_numbers", 1).values == [1.0]
    assert store.versions("toy_numbers") == [1, 2]
    assert store.heads() == {"toy_numbers": 2}


def test_identical_payload_is_deduplicated(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.put("toy_numbers", ToyNumbers(values=[1.0]), producer=P)
    assert store.put("toy_numbers", {"values": [1.0]}, producer=P) == 1
    assert store.put("toy_numbers", ToyNumbers(values=[1.0]), producer=P, dedupe=False) == 2


def test_same_commit_token_is_idempotent(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    token = Producer(stage="s", agent="a", token="run:s:1")
    assert store.put("toy_numbers", ToyNumbers(values=[1.0]), producer=token) == 1
    assert store.put("toy_numbers", ToyNumbers(values=[9.0]), producer=token) == 1  # re-commit after crash
    assert store.version_with_token("toy_numbers", "run:s:1") == 1
    assert store.version_with_token("toy_numbers", "other") is None


def test_validation_and_unknown_types(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ArtifactError, match="expects ToyNumbers"):
        store.put("toy_numbers", ToyStats(mean=1, count=1), producer=P)
    with pytest.raises(ArtifactError, match="does not match"):
        store.put("toy_numbers", {"values": "not a list"}, producer=P)
    with pytest.raises(ArtifactError, match="no schema registered"):
        store.put("nobody_registered_this", {"x": 1}, producer=P)
    with pytest.raises(ArtifactError, match="does not exist"):
        store.get("toy_stats")


def test_staging_respects_declared_outputs_and_commits(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    staged = StagedOutputs(store, "analyze", allowed={"toy_stats"})
    with pytest.raises(ArtifactError, match="may not produce"):
        staged.put("toy_numbers", ToyNumbers(values=[1.0]))
    staged.put("toy_stats", ToyStats(mean=1.5, count=2))
    assert staged.has("toy_stats") and staged.get("toy_stats").mean == 1.5
    assert store.head("toy_stats") is None  # invisible until committed
    versions = staged.commit(Producer(stage="analyze", agent="x", token="t1"), inputs={"toy_numbers": 3})
    assert versions == {"toy_stats": 1}
    assert store.envelope("toy_stats").inputs == {"toy_numbers": 3}
    assert staged.names() == []


def test_commit_resumes_after_partial_commit(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    staged = StagedOutputs(store, "s")
    staged.put("toy_numbers", ToyNumbers(values=[1.0]))
    staged.put("toy_stats", ToyStats(mean=1.0, count=1))
    producer = Producer(stage="s", agent="a", token="tok")
    # simulate a crash after the first artifact was published
    store.put("toy_numbers", ToyNumbers(values=[1.0]), producer=producer)
    (staged.dir / "toy_numbers.json").unlink()
    assert staged.commit(producer, {}) == {"toy_stats": 1}
    assert store.heads() == {"toy_numbers": 1, "toy_stats": 1}
