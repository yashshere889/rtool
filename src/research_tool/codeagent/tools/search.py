"""Search tools: glob (find files by name) and grep (search contents, ripgrep when available)."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import shutil
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..tool import PermissionDecision, Tool, ToolContext, ToolError, ToolInput

IGNORED_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".agent",
    "dist",
    "build",
}


def _base_dir(ctx: ToolContext, path: str | None) -> Path:
    base = ctx.resolve(path or ".")
    if not base.exists():
        raise ToolError(f"Path not found: {path}")
    return base


def _outside_check(ctx: ToolContext, path: str | None) -> PermissionDecision:
    p = ctx.resolve(path or ".")
    if not ctx.freely_readable(p):
        return PermissionDecision.ask(f"searching {p}, which is outside the workspace")
    return PermissionDecision.passthrough()


class GlobInput(ToolInput):
    pattern: str = Field(description="Glob pattern, e.g. '**/*.py' or 'src/**/test_*.ts'")
    path: str | None = Field(None, description="Directory to search in; defaults to the working directory")


class GlobTool(Tool[GlobInput]):
    name = "glob"
    description = (
        "Find files by name pattern. '**' matches any number of directories, so use '**/*.py' "
        "for all Python files. Returns paths sorted by most recently modified."
    )
    Input = GlobInput
    timeout_s = 60

    def is_read_only(self, inp: GlobInput) -> bool:
        return True

    def is_concurrency_safe(self, inp: GlobInput) -> bool:
        return True

    async def check_permissions(self, inp: GlobInput, ctx: ToolContext) -> PermissionDecision:
        return _outside_check(ctx, inp.path)

    async def call(self, inp: GlobInput, ctx: ToolContext) -> str:
        base = _base_dir(ctx, inp.path)
        if not base.is_dir():
            raise ToolError(f"{inp.path} is not a directory.")

        def find() -> list[Path]:
            hits = []
            for p in base.glob(inp.pattern):
                rel_parts = p.relative_to(base).parts
                if any(part in IGNORED_DIRS for part in rel_parts[:-1]) or not p.is_file():
                    continue
                hits.append(p)
            return hits

        hits = await asyncio.to_thread(find)
        if not hits:
            return f"No files match {inp.pattern!r}."
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        limit = 200
        lines = [ctx.display(p) for p in hits[:limit]]
        more = f"\n[{len(hits) - limit} more not shown; narrow the pattern.]" if len(hits) > limit else ""
        return "\n".join(lines) + more


class GrepInput(ToolInput):
    pattern: str = Field(description="Regular expression to search for")
    path: str | None = Field(None, description="File or directory to search; defaults to the working directory")
    glob: str | None = Field(None, description="Only search files whose name matches this glob, e.g. '*.py'")
    ignore_case: bool = False
    output: Literal["files", "content", "count"] = Field(
        "files", description="'files' lists matching files, 'content' shows matching lines, 'count' counts per file"
    )
    context_lines: int = Field(0, ge=0, le=10, description="Lines of context around each match (content output only)")
    max_results: int = Field(200, ge=1, le=2000)


class GrepTool(Tool[GrepInput]):
    name = "grep"
    description = (
        "Search file contents with a regular expression (ripgrep syntax). Use output='files' to find "
        "where something is, then output='content' to see the lines. Skips .git, node_modules and similar."
    )
    Input = GrepInput
    timeout_s = 60

    def is_read_only(self, inp: GrepInput) -> bool:
        return True

    def is_concurrency_safe(self, inp: GrepInput) -> bool:
        return True

    async def check_permissions(self, inp: GrepInput, ctx: ToolContext) -> PermissionDecision:
        return _outside_check(ctx, inp.path)

    async def validate(self, inp: GrepInput, ctx: ToolContext) -> str | None:
        try:
            re.compile(inp.pattern)
        except re.error as exc:
            # ripgrep's syntax is close to Python's; this catches most mistakes early
            return f"Invalid regular expression {inp.pattern!r}: {exc}"
        return None

    async def call(self, inp: GrepInput, ctx: ToolContext) -> str:
        base = _base_dir(ctx, inp.path)
        rg = shutil.which("rg")
        lines = await (
            self._ripgrep(rg, inp, base, ctx) if rg else asyncio.to_thread(self._python_grep, inp, base, ctx)
        )
        if not lines:
            return "No matches."
        shown = lines[: inp.max_results]
        more = (
            f"\n[{len(lines) - len(shown)} more results not shown; narrow the search.]"
            if len(lines) > len(shown)
            else ""
        )
        return "\n".join(shown) + more

    async def _ripgrep(self, rg: str, inp: GrepInput, base: Path, ctx: ToolContext) -> list[str]:
        args = [rg, "--no-heading", "--color=never", "--no-messages"]
        for d in IGNORED_DIRS:
            args += ["--glob", f"!{d}"]
        if inp.ignore_case:
            args.append("-i")
        if inp.glob:
            args += ["--glob", inp.glob]
        if inp.output == "files":
            args.append("-l")
        elif inp.output == "count":
            args.append("-c")
        else:
            args.append("-n")
            if inp.context_lines:
                args += ["-C", str(inp.context_lines)]
        target = os.path.relpath(base, ctx.cwd) if base.is_relative_to(ctx.cwd) else str(base)
        args += ["--", inp.pattern] + ([] if target == "." else [target])
        proc = await asyncio.create_subprocess_exec(
            *args, cwd=ctx.cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            out, err = await proc.communicate()
        except asyncio.CancelledError:
            proc.kill()
            raise
        if proc.returncode == 1:
            return []
        if proc.returncode not in (0, None):
            raise ToolError(f"grep failed: {err.decode(errors='replace').strip()}")
        return out.decode(errors="replace").splitlines()

    def _python_grep(self, inp: GrepInput, base: Path, ctx: ToolContext) -> list[str]:
        regex = re.compile(inp.pattern, re.IGNORECASE if inp.ignore_case else 0)
        files = [base] if base.is_file() else []
        if base.is_dir():
            for root, dirs, names in os.walk(base):
                dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
                files += [Path(root) / n for n in sorted(names) if not inp.glob or fnmatch.fnmatch(n, inp.glob)]
        results: list[str] = []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            hits = [(i, line) for i, line in enumerate(text.splitlines(), 1) if regex.search(line)]
            if not hits:
                continue
            name = ctx.display(f)
            if inp.output == "files":
                results.append(name)
            elif inp.output == "count":
                results.append(f"{name}:{len(hits)}")
            else:
                results += [f"{name}:{i}:{line}" for i, line in hits]
            if len(results) > inp.max_results:
                break
        return results
