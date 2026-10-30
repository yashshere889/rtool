"""What the local demo needs on Windows, checked where it can be without Windows."""

from __future__ import annotations

import mimetypes
from pathlib import Path, PureWindowsPath

from research_tool.agents.coder.env import windows_job_command
from research_tool.knowledge.store import KnowledgeStore
from research_tool.web.app import create_app  # noqa: F401 - importing it registers the types


def test_the_array_task_command_is_written_for_cmd() -> None:
    code = PureWindowsPath(r"C:\Users\A B\rtool-runs\p\r\work\code")
    command = windows_job_command(code, r"C:\Py 3\python.exe", "batches/full-1.json")  # type: ignore[arg-type]
    assert command == (
        r'cd /d "C:\Users\A B\rtool-runs\p\r\work\code" && "C:\Py 3\python.exe" launch.py '
        r'--batch "batches/full-1.json" --task %SLURM_ARRAY_TASK_ID%'
    )


def test_scripts_are_served_as_scripts_whatever_the_registry_says() -> None:
    assert mimetypes.guess_type("index.js")[0] == "text/javascript"
    assert mimetypes.guess_type("index.css")[0] == "text/css"


def test_a_read_only_store_opens_from_a_path_with_uri_characters(tmp_path: Path) -> None:
    path = tmp_path / "a b#c%d" / "kb.sqlite"
    with KnowledgeStore(path):
        pass
    KnowledgeStore(path, readonly=True).close()
