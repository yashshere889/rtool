from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_tool.core.errors import LeaseHeld
from research_tool.core.fs import FileLease, append_jsonl, atomic_write_json, read_json


def test_atomic_write_roundtrip_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b.json"
    atomic_write_json(target, {"x": 1, "y": [1, 2]})
    atomic_write_json(target, {"x": 2})
    assert read_json(target) == {"x": 2}
    assert [p.name for p in target.parent.iterdir()] == ["b.json"]


def test_append_jsonl(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    append_jsonl(log, {"n": 1})
    append_jsonl(log, {"n": 2})
    assert [json.loads(line)["n"] for line in log.read_text().splitlines()] == [1, 2]


def test_lease_is_exclusive_until_released(tmp_path: Path) -> None:
    path = tmp_path / "run.lease"
    first, second = FileLease(path, owner="a"), FileLease(path, owner="b")
    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()
    assert not path.exists()


def test_expired_lease_can_be_taken_over(tmp_path: Path) -> None:
    path = tmp_path / "run.lease"
    now = [1000.0]
    dead = FileLease(path, ttl=10, owner="dead", clock=lambda: now[0])
    assert dead.acquire()
    alive = FileLease(path, ttl=10, owner="alive", clock=lambda: now[0])
    assert not alive.acquire()
    now[0] += 11  # the dead holder never renewed
    assert alive.acquire()
    assert alive.holder().owner == "alive"  # type: ignore[union-attr]
    with pytest.raises(LeaseHeld):
        dead.renew()  # the old holder notices it lost the lease


def test_requeued_job_reclaims_its_own_lease(tmp_path: Path) -> None:
    path = tmp_path / "run.lease"
    assert FileLease(path, owner="ep-77").acquire()  # killed by preemption: never released
    assert not FileLease(path, owner="ep-78").acquire()  # someone else still has to wait
    again = FileLease(path, owner="ep-77")  # the same Slurm job, requeued
    assert again.acquire() and again.holder().owner == "ep-77"  # type: ignore[union-attr]


def test_context_manager_raises_when_held(tmp_path: Path) -> None:
    path = tmp_path / "run.lease"
    with FileLease(path, owner="a"), pytest.raises(LeaseHeld), FileLease(path, owner="b"):
        pass
