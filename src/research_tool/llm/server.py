"""Start and stop a vLLM server on the current node (used inside an episode job).

The server listens on 127.0.0.1 only, with a random port and a random bearer
token passed through the environment (``VLLM_API_KEY``), so other users on a
shared GPU node can neither see the token in ``ps`` nor reach the server from
another machine.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import signal
import socket
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..core.config import LLMConfig
from ..core.errors import RToolError
from ..core.fs import atomic_write_json


class VLLMServerError(RToolError):
    """The vLLM server could not be started or did not become healthy."""


def expand_path(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def log_tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except FileNotFoundError:
        return "(no log written)"


@dataclass
class ServerHandle:
    url: str  # e.g. http://127.0.0.1:41234  (the OpenAI API is under /v1)
    api_key: str
    pid: int
    log_path: Path
    process: subprocess.Popen[bytes]


class VLLMServer:
    def __init__(
        self,
        llm: LLMConfig,
        log_dir: Path,
        *,
        command_override: Callable[[int], list[str]] | None = None,
        poll_interval_s: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.llm = llm
        self.settings = llm.server
        self.log_dir = Path(log_dir)
        self._command_override = command_override
        self.poll_interval_s = poll_interval_s
        self.clock = clock

    def command(self, port: int) -> list[str]:
        if self._command_override is not None:
            return self._command_override(port)
        s = self.settings
        if not s.container:
            raise VLLMServerError(
                "llm.server.container is not set: point it at an Apptainer image with vLLM "
                "(e.g. '${NEMOTRON_SIF}' after `module load nemotron/nano-12b-v2`, if that image has vLLM)"
            )
        container = os.path.expandvars(s.container)
        if "$" in container:
            raise VLLMServerError(f"llm.server.container references an unset variable: {s.container}")
        args = ["apptainer", "exec", "--nv"]
        if s.home:
            args += ["--home", expand_path(s.home)]
        for bind in s.binds:
            args += ["--bind", os.path.expandvars(bind)]
        args += [
            container,
            "vllm",
            "serve",
            self.llm.model,
            "--host",
            s.host,
            "--port",
            str(port),
            "--served-model-name",
            self.llm.served_model,
            "--max-model-len",
            str(s.max_model_len),
            "--max-num-seqs",
            str(s.max_num_seqs),
            "--gpu-memory-utilization",
            str(s.gpu_memory_utilization),
        ]
        if s.reasoning_parser:
            args += ["--reasoning-parser", s.reasoning_parser]
        return args + list(s.extra_args)

    def environment(self, api_key: str) -> dict[str, str]:
        # The container's Python must see only its own packages: nothing from the episode's
        # venv (PYTHONPATH) and nothing from ~/.local, which the container would otherwise import.
        inner = {"PYTHONNOUSERSITE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "VLLM_API_KEY": api_key}
        if self.settings.hf_home:
            inner["HF_HOME"] = expand_path(self.settings.hf_home)
        if self.settings.offline:
            inner["HF_HUB_OFFLINE"] = "1"
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env.update(inner)
        # Images may set these themselves (the Nemotron image sets HF_HOME), and an image's
        # own values beat the host's; APPTAINERENV_* values beat the image's.
        env.update({f"APPTAINERENV_{k}": v for k, v in inner.items()})
        return env

    def start(self) -> ServerHandle:
        port = self.settings.port or free_port(self.settings.host)
        api_key = secrets.token_urlsafe(24)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.settings.home:
            Path(expand_path(self.settings.home)).mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"vllm-{port}.log"
        command = self.command(port)
        with open(log_path, "ab") as log:
            log.write(f"$ {' '.join(command)}\n".encode())
            log.flush()
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=self.environment(api_key),
                start_new_session=True,  # own process group, so stop() can kill apptainer + vllm together
            )
        handle = ServerHandle(
            url=f"http://{self.settings.host}:{port}",
            api_key=api_key,
            pid=process.pid,
            log_path=log_path,
            process=process,
        )
        try:
            self._wait_until_healthy(handle)
        except BaseException:
            self.stop(handle, grace_s=5)
            raise
        atomic_write_json(
            self.log_dir / "llm_endpoint.json", {"url": handle.url, "pid": handle.pid, "log": str(log_path)}
        )
        return handle

    def _wait_until_healthy(self, handle: ServerHandle) -> None:
        deadline = self.clock() + self.settings.startup_timeout_s
        headers = {"Authorization": f"Bearer {handle.api_key}"}
        with httpx.Client(timeout=5.0, headers=headers) as client:
            while True:
                code = handle.process.poll()
                if code is not None:
                    raise VLLMServerError(
                        f"vLLM exited with code {code} during startup; last log lines:\n{log_tail(handle.log_path)}"
                    )
                with contextlib.suppress(httpx.TransportError):
                    if client.get(handle.url + "/health").status_code == 200:
                        return
                if self.clock() > deadline:
                    raise VLLMServerError(
                        f"vLLM not healthy after {self.settings.startup_timeout_s:.0f}s; "
                        f"last log lines:\n{log_tail(handle.log_path)}"
                    )
                time.sleep(self.poll_interval_s)

    @staticmethod
    def stop(handle: ServerHandle, grace_s: float = 30.0) -> None:
        if handle.process.poll() is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(handle.process.pid, signal.SIGTERM)
        try:
            handle.process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(handle.process.pid, signal.SIGKILL)
            handle.process.wait(timeout=10)

    @contextlib.contextmanager
    def running(self) -> Iterator[ServerHandle]:
        handle = self.start()
        try:
            yield handle
        finally:
            self.stop(handle)
