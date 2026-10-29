"""vLLM server manager (real subprocesses with a tiny fake server) and the prompt library."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

from research_tool.core.config import LLMConfig, LLMServerConfig
from research_tool.llm.prompts import PromptError, PromptLibrary
from research_tool.llm.server import VLLMServer, VLLMServerError

FAKE_SERVER = """
import http.server, os, sys, time
port = int(sys.argv[1])
print("fake vllm starting; key set:", bool(os.environ.get("VLLM_API_KEY")), flush=True)
time.sleep(0.3)
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def config(**server: object) -> LLMConfig:
    return LLMConfig(server=LLMServerConfig(**server))  # type: ignore[arg-type]


def test_command_follows_the_model_card(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SIF_DIR", "/images")
    cfg = config(container="${SIF_DIR}/vllm.sif", binds=["/mnt/scratch:/mnt/scratch"], hf_home="~/hf")
    server = VLLMServer(cfg, tmp_path)
    cmd = server.command(4321)
    assert cmd[:5] == ["apptainer", "exec", "--nv", "--bind", "/mnt/scratch:/mnt/scratch"]
    assert cmd[5:9] == ["/images/vllm.sif", "vllm", "serve", "nvidia/NVIDIA-Nemotron-Nano-12B-v2"]
    joined = " ".join(cmd)
    for expected in (
        "--host 127.0.0.1",
        "--port 4321",
        "--served-model-name nemotron",
        "--trust-remote-code",
        "--mamba_ssm_cache_dtype float32",
        "--reasoning-parser deepseek_r1",
    ):
        assert expected in joined
    env = server.environment("tok")
    assert env["VLLM_API_KEY"] == "tok" and env["HF_HUB_OFFLINE"] == "1"
    assert env["HF_HOME"] == env["APPTAINERENV_HF_HOME"] == str(Path("~/hf").expanduser())  # beats the image's own
    assert env["APPTAINERENV_HF_HUB_OFFLINE"] == "1" and env["APPTAINERENV_VLLM_API_KEY"] == "tok"
    assert "tok" not in joined  # the key never appears on the command line (visible in ps)


def test_container_gets_its_own_home_and_clean_python(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PYTHONPATH", "/venv/lib")
    server = VLLMServer(config(container="/images/vllm.sif", home=str(tmp_path / "vhome")), tmp_path)
    cmd = server.command(1)
    assert cmd[3:6] == ["--home", str(tmp_path / "vhome"), "/images/vllm.sif"]
    env = server.environment("tok")
    assert "PYTHONPATH" not in env and env["PYTHONNOUSERSITE"] == "1" and env["VLLM_NO_USAGE_STATS"] == "1"


def test_container_must_be_configured(tmp_path: Path) -> None:
    with pytest.raises(VLLMServerError, match="container is not set"):
        VLLMServer(config(), tmp_path).command(1)
    with pytest.raises(VLLMServerError, match="unset variable"):
        VLLMServer(config(container="${RTOOL_SURELY_UNSET}/x.sif"), tmp_path).command(1)


def test_start_wait_and_stop(tmp_path: Path) -> None:
    script = tmp_path / "fake_vllm.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    server = VLLMServer(
        config(startup_timeout_s=30),
        tmp_path / "logs",
        command_override=lambda port: [sys.executable, str(script), str(port)],
        poll_interval_s=0.1,
    )
    with server.running() as handle:
        assert httpx.get(handle.url + "/health").status_code == 200
        assert (tmp_path / "logs" / "llm_endpoint.json").exists()
        process = handle.process
    assert process.poll() is not None  # stopped on exit
    assert "key set: True" in handle.log_path.read_text()


def test_startup_crash_reports_the_log(tmp_path: Path) -> None:
    crash = [sys.executable, "-c", "print('CUDA error: out of memory', flush=True); raise SystemExit(3)"]
    server = VLLMServer(config(), tmp_path, command_override=lambda _p: crash, poll_interval_s=0.05)
    with pytest.raises(VLLMServerError, match=r"(?s)exited with code 3.*out of memory"):
        server.start()


def test_startup_timeout_kills_the_process(tmp_path: Path) -> None:
    hang = [sys.executable, "-c", "import time; time.sleep(60)"]
    server = VLLMServer(config(startup_timeout_s=0.3), tmp_path, command_override=lambda _p: hang, poll_interval_s=0.05)
    with pytest.raises(VLLMServerError, match="not healthy"):
        server.start()


def test_prompt_library_overrides_and_strictness(tmp_path: Path) -> None:
    own, pack = tmp_path / "own", tmp_path / "pack"
    own.mkdir()
    pack.mkdir()
    (own / "extract.j2").write_text("Extract from {{ paper.title }}.\n", encoding="utf-8")
    (own / "only_own.j2").write_text("{% for q in questions %}- {{ q }}\n{% endfor %}", encoding="utf-8")
    (pack / "extract.j2").write_text("Pack version for {{ paper.title }}.\n", encoding="utf-8")
    library = PromptLibrary([pack, own])
    rendered = library.render("extract.j2", paper={"title": "T"})
    assert rendered.text == "Pack version for T.\n" and rendered.source.endswith("pack/extract.j2")
    assert rendered.template_hash.startswith("sha256:") and set(rendered.provenance()) == {
        "prompt",
        "prompt_hash",
        "prompt_source",
    }
    assert library.render("only_own.j2", questions=["a", "b"]).text == "- a\n- b\n"
    assert library.names() == ["extract.j2", "only_own.j2"]
    with pytest.raises(PromptError, match="undefined"):
        library.render("extract.j2")
    with pytest.raises(PromptError, match="not found"):
        library.render("missing.j2")
    assert PromptLibrary.for_agent("research_tool.testing", "toy").names() == []  # no prompts dir: empty


def test_startup_crash_names_the_root_cause_even_when_the_tail_does_not(tmp_path: Path) -> None:
    """2026-10-15 on gpu45: the cause sat 70 lines above the end, under ANSI colour codes."""
    log = "\n".join(
        [
            "INFO 10-15 09:29:43 [core.py:77] Initializing a V1 LLM engine (v0.11.0)",
            "\x1b[1;36m(EngineCore_DP0 pid=2663372)\x1b[0;0m ERROR [core.py:708] "
            "OSError: [Errno 28] No space left on device",
            *[f'  File "/usr/lib/python3.10/x{i}.py", line {i}, in f' for i in range(60)],
            "(APIServer pid=2662695) RuntimeError: Engine core initialization failed. See root cause above.",
        ]
    )
    script = tmp_path / "crash.py"
    script.write_text(f"print({log!r}, flush=True)\nraise SystemExit(1)\n")
    crash = [sys.executable, str(script)]
    server = VLLMServer(config(), tmp_path, command_override=lambda _p: crash, poll_interval_s=0.05)
    with pytest.raises(VLLMServerError) as caught:
        server.start()
    first_line = str(caught.value).split(";")[0]
    assert first_line == "vLLM exited with code 1 during startup (OSError: [Errno 28] No space left on device)"
