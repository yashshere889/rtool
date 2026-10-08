from __future__ import annotations

from pathlib import Path

import pytest

from research_tool.core.config import GateMode, load_config, parse_set_overrides
from research_tool.core.errors import ConfigError


def test_defaults_include_nemotron_profiles() -> None:
    cfg = load_config(env={})
    assert cfg.llm.model == "nvidia/NVIDIA-Nemotron-Nano-12B-v2"
    assert cfg.llm.profile("extract").reasoning == "off"
    assert cfg.llm.profile("reason").temperature == pytest.approx(0.6)
    with pytest.raises(ConfigError):
        cfg.llm.profile("nope")


def test_partial_profile_override_keeps_other_fields(tmp_path: Path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("llm:\n  profiles:\n    extract:\n      max_tokens: 99\n", encoding="utf-8")
    cfg = load_config(f, env={})
    assert cfg.llm.profile("extract").max_tokens == 99
    assert cfg.llm.profile("extract").reasoning == "off"
    assert "judge" in cfg.llm.profiles


def test_env_interpolation_and_missing_variable(tmp_path: Path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("paths:\n  runs_root: ${SCRATCH}/runs\n", encoding="utf-8")
    assert str(load_config(f, env={"SCRATCH": "/mnt/scratch/me"}).paths.runs_root) == "/mnt/scratch/me/runs"
    with pytest.raises(ConfigError, match="SCRATCH"):
        load_config(f, env={})
    g = tmp_path / "d.yaml"
    g.write_text("paths:\n  runs_root: ${SCRATCH:-/tmp/x}\n", encoding="utf-8")
    assert str(load_config(g, env={}).paths.runs_root) == "/tmp/x"


def test_unknown_keys_fail_loudly(tmp_path: Path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("revisions:\n  max_round: 3\n", encoding="utf-8")  # typo
    with pytest.raises(ConfigError, match="max_round"):
        load_config(f, env={})


def test_set_overrides_are_typed() -> None:
    parsed = parse_set_overrides(["revisions.max_rounds=3", "gates.default=auto", "llm.profiles.write.temperature=0.2"])
    cfg = load_config(overrides=parsed, env={})
    assert cfg.revisions.max_rounds == 3
    assert cfg.gates.mode("anything") is GateMode.AUTO
    assert cfg.llm.profile("write").temperature == pytest.approx(0.2)
    with pytest.raises(ConfigError):
        parse_set_overrides(["no-equals-sign"])


def test_example_config_file_is_valid() -> None:
    example = Path(__file__).resolve().parents[2] / "configs" / "example.yaml"
    cfg = load_config(example, env={})
    assert cfg.gates.mode("gpu_runs") is GateMode.ASK
    assert cfg.llm.profile("judge").samples == 3


def test_gate_modes_per_gate() -> None:
    cfg = load_config(overrides={"gates": {"modes": {"final": "auto"}}}, env={})
    assert cfg.gates.mode("final") is GateMode.AUTO
    assert cfg.gates.mode("plan") is GateMode.ASK
