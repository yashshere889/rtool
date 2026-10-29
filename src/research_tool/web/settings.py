"""How a UI server was started."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

STATIC_DIR = Path(__file__).parent / "static"

Mode = Literal["local", "login", "job"]


class UISettings(BaseModel):
    runs_root: Path
    token: str = Field(min_length=16)
    # local: this machine, episodes run inside the server with the demo fakes (development, demos)
    # login / job: on the cluster; decisions submit episode jobs to Slurm, like the CLI
    mode: Mode = "local"
    config_files: list[Path] = Field(default_factory=list)
    static_dir: Path = STATIC_DIR

    @property
    def runs_locally(self) -> bool:
        return self.mode == "local"
