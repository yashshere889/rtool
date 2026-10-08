from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from research_tool.core.config import load_config
from research_tool.hpc.doctor import _internet, run_doctor
from research_tool.hpc.storage import Quota, QuotaError, ensure_room, mount_for, parse_lfs_quota

LFS = """Disk quotas for usr someone (uid 12345):
     Filesystem  kbytes   quota   limit   grace   files   quota   limit   grace
   /mnt/scratch 1048576       0 2147483648       -   12000       0  300000       -
"""
LFS_WRAPPED_OVER = """Disk quotas for usr someone (uid 12345):
     Filesystem  kbytes   quota   limit   grace   files   quota   limit   grace
/mnt/fastscratch
                524288000*      0 524288000       -  10       0  500000       -
"""


def test_parse_lfs_quota() -> None:
    q = parse_lfs_quota(LFS, "/mnt/scratch")
    assert q.used_bytes == 1024**3 and q.limit_bytes == 2 * 1024**4
    assert (q.used_files, q.limit_files) == (12000, 300000)
    wrapped = parse_lfs_quota(LFS_WRAPPED_OVER, "/mnt/fastscratch")  # long name wraps; '*' = over quota
    assert wrapped.used_bytes == wrapped.limit_bytes and wrapped.limit_files == 500000
    with pytest.raises(QuotaError):
        parse_lfs_quota("nothing useful", "/mnt/scratch")


def test_ensure_room() -> None:
    quota = Quota("/mnt/scratch", used_bytes=10, limit_bytes=100, used_files=10, limit_files=100)
    check = lambda mount: quota  # noqa: E731
    assert ensure_room(Path("/mnt/scratch/u/x"), extra_bytes=50, quota_fn=check) is quota
    with pytest.raises(QuotaError, match="space quota"):
        ensure_room(Path("/mnt/scratch/u/x"), extra_bytes=85, quota_fn=check)
    with pytest.raises(QuotaError, match="file-count quota"):
        ensure_room(Path("/mnt/scratch/u/x"), extra_files=85, quota_fn=check)
    assert ensure_room(Path("/home/u"), extra_bytes=10**15, quota_fn=check) is None  # not quota-managed
    assert mount_for(Path("/mnt/fastscratch/u"), ["/mnt/scratch", "/mnt/fastscratch"]) == "/mnt/fastscratch"


def test_doctor_locally_with_fakes() -> None:
    checks = {c.name: c for c in run_doctor(load_config(env={}), network=False)}
    assert checks["sqlite FTS5"].status == "ok"
    assert checks["slurm"].status == "skip" and checks["llm"].status == "warn"


def test_doctor_vllm_checks_report_missing_pieces(tmp_path: Path) -> None:
    cfg = load_config(
        overrides={
            "llm": {"backend": "vllm", "server": {"container": str(tmp_path / "none.sif"), "hf_home": str(tmp_path)}}
        },
        env={},
    )
    checks = {c.name: c for c in run_doctor(cfg, network=False, module_env=lambda _m: None)}
    assert checks["vLLM image"].status == "fail" and "does not exist" in checks["vLLM image"].detail
    assert checks["model weights"].status == "fail"


def test_doctor_loads_the_server_modules_like_an_episode(tmp_path: Path) -> None:
    sif = tmp_path / "nemotron.sif"
    sif.write_bytes(b"")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "apptainer").write_text("#!/bin/sh\n", encoding="utf-8")
    (bin_dir / "apptainer").chmod(0o755)
    weights = tmp_path / "hf" / "hub" / "models--nvidia--NVIDIA-Nemotron-Nano-12B-v2"
    weights.mkdir(parents=True)
    cfg = load_config(
        overrides={
            "llm": {
                "backend": "vllm",
                "server": {"container": "$FAKE_SIF", "modules": ["nemotron/x"], "hf_home": str(tmp_path / "hf")},
            }
        },
        env={},
    )
    seen: list[list[str]] = []

    def fake_module_env(modules: list[str]) -> dict[str, str]:
        seen.append(modules)
        return {"FAKE_SIF": str(sif), "PATH": str(bin_dir)}

    checks = {c.name: c for c in run_doctor(cfg, network=False, module_env=fake_module_env)}
    assert seen == [["nemotron/x"]]
    assert checks["vLLM image"].status == "ok" and "after module load nemotron/x" in checks["vLLM image"].detail
    assert "apptainer" not in checks and checks["model weights"].status == "ok"


def test_internet_checks() -> None:
    def fake_get(url: str) -> int:
        if "arxiv" in url:
            raise httpx.ConnectError("blocked")
        return 200

    results = {c.name: c.status for c in _internet(fake_get)}
    assert results["reach arXiv"] == "fail" and results["reach OpenAlex"] == "ok"
