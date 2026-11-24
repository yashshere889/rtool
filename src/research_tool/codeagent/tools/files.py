"""File tools: read_file, write_file, edit_file.

Safety properties:
* Paths outside the workspace need approval (``check_permissions`` returns ``ask``).
* You must ``read_file`` an existing file before writing or editing it, and it must not have
  changed on disk since. This stops the model from editing a file it never looked at, or
  clobbering changes made by the user or a formatter in the meantime.
* ``edit_file`` replaces an exact string that must occur exactly once (or ``replace_all``),
  which is far more reliable for LLMs than line-number or diff based edits.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from ..tool import PermissionDecision, Tool, ToolContext, ToolError, ToolInput, ToolResult

MAX_LINE_CHARS = 2000


def _read_text(p: Path) -> str:
    with open(p, encoding="utf-8", errors="replace", newline="") as fh:
        return fh.read()


def _write_text(p: Path, text: str) -> None:
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def _stale_read_error(p: Path, ctx: ToolContext) -> str | None:
    if not p.exists():
        return None
    seen = ctx.read_files.get(p)
    if seen is None:
        return f"Read {ctx.display(p)} with read_file before modifying it."
    if p.stat().st_mtime_ns != seen:
        return f"{ctx.display(p)} changed on disk since you last read it. Read it again before modifying it."
    return None


def _numbered(lines: list[str], first_line_no: int) -> str:
    out = []
    for i, line in enumerate(lines):
        if len(line) > MAX_LINE_CHARS:
            line = line[:MAX_LINE_CHARS] + " …[line truncated]"
        out.append(f"{first_line_no + i:>6}\t{line}")
    return "\n".join(out)


# --------------------------------------------------------------------------------------------
class ReadFileInput(ToolInput):
    path: str = Field(description="File path, absolute or relative to the working directory")
    offset: int = Field(1, ge=1, description="1-based line number to start reading from")
    limit: int = Field(2000, ge=1, le=5000, description="Maximum number of lines to return")


class ReadFileTool(Tool[ReadFileInput]):
    name = "read_file"
    description = (
        "Read a text file. Each line is returned as '<line number>\\t<text>'. Read a file before "
        "editing it. For large files, use offset and limit to read a section at a time."
    )
    Input = ReadFileInput
    max_result_chars = 60_000

    def is_read_only(self, inp: ReadFileInput) -> bool:
        return True

    def is_concurrency_safe(self, inp: ReadFileInput) -> bool:
        return True

    def describe_call(self, inp: ReadFileInput) -> str:
        return f"read_file {inp.path}"

    async def check_permissions(self, inp: ReadFileInput, ctx: ToolContext) -> PermissionDecision:
        p = ctx.resolve(inp.path)
        if not ctx.freely_readable(p):
            return PermissionDecision.ask(f"reading {p}, which is outside the workspace")
        return PermissionDecision.passthrough()

    def permission_subjects(self, inp: ReadFileInput) -> list[str]:
        return [inp.path]

    async def call(self, inp: ReadFileInput, ctx: ToolContext) -> str:
        p = ctx.resolve(inp.path)
        if not p.exists():
            raise ToolError(f"File not found: {inp.path}")
        if p.is_dir():
            raise ToolError(f"{inp.path} is a directory. Use glob to list files.")
        mtime = p.stat().st_mtime_ns
        data = p.read_bytes()
        if b"\0" in data[:8192]:
            raise ToolError(f"{inp.path} looks like a binary file; not reading it.")
        lines = data.decode("utf-8", errors="replace").splitlines()
        ctx.read_files[p] = mtime
        total = len(lines)
        if total == 0:
            return "(empty file)"
        start = inp.offset - 1
        if start >= total:
            raise ToolError(f"offset {inp.offset} is past the end of the file ({total} lines).")
        chunk = lines[start : start + inp.limit]
        text = _numbered(chunk, inp.offset)
        end = start + len(chunk)
        if end < total or start > 0:
            text += f"\n\n[Lines {inp.offset}-{end} of {total}."
            text += f" Use offset={end + 1} to continue.]" if end < total else "]"
        return text


# --------------------------------------------------------------------------------------------
class WriteFileInput(ToolInput):
    path: str = Field(description="File path, absolute or relative to the working directory")
    content: str = Field(description="The complete new content of the file")


class WriteFileTool(Tool[WriteFileInput]):
    name = "write_file"
    description = (
        "Create a file, or replace a file's entire content. Prefer edit_file for changes to an "
        "existing file. An existing file must have been read with read_file first."
    )
    Input = WriteFileInput

    def is_file_edit(self, inp: WriteFileInput) -> bool:
        return True

    def describe_call(self, inp: WriteFileInput) -> str:
        return f"write_file {inp.path} ({len(inp.content.splitlines())} lines)"

    def permission_subjects(self, inp: WriteFileInput) -> list[str]:
        return [inp.path]

    async def validate(self, inp: WriteFileInput, ctx: ToolContext) -> str | None:
        p = ctx.resolve(inp.path)
        if p.is_dir():
            return f"{inp.path} is a directory."
        return _stale_read_error(p, ctx)

    async def check_permissions(self, inp: WriteFileInput, ctx: ToolContext) -> PermissionDecision:
        p = ctx.resolve(inp.path)
        if not ctx.in_workspace(p):
            return PermissionDecision.ask(f"writing {p}, which is outside the workspace")
        return PermissionDecision.passthrough()

    async def call(self, inp: WriteFileInput, ctx: ToolContext) -> ToolResult:
        p = ctx.resolve(inp.path)
        existed = p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        _write_text(p, inp.content)
        ctx.read_files[p] = p.stat().st_mtime_ns
        verb = "Updated" if existed else "Created"
        return ToolResult(f"{verb} {ctx.display(p)} ({len(inp.content.splitlines())} lines).")


# --------------------------------------------------------------------------------------------
class EditFileInput(ToolInput):
    path: str = Field(description="File path, absolute or relative to the working directory")
    old_string: str = Field(
        description="Exact text to replace, including indentation. Must be unique in the file unless replace_all is true"
    )
    new_string: str = Field(description="Text to put in its place")
    replace_all: bool = Field(False, description="Replace every occurrence instead of exactly one")


class EditFileTool(Tool[EditFileInput]):
    name = "edit_file"
    description = (
        "Replace an exact string in a file. old_string must match the file exactly (whitespace "
        "included, without the line-number prefix from read_file) and occur exactly once, unless "
        "replace_all is true. Include enough surrounding lines to make it unique. Read the file first."
    )
    Input = EditFileInput

    def is_file_edit(self, inp: EditFileInput) -> bool:
        return True

    def describe_call(self, inp: EditFileInput) -> str:
        return f"edit_file {inp.path}"

    def permission_subjects(self, inp: EditFileInput) -> list[str]:
        return [inp.path]

    async def validate(self, inp: EditFileInput, ctx: ToolContext) -> str | None:
        p = ctx.resolve(inp.path)
        if not p.exists():
            return f"File not found: {inp.path}. Use write_file to create a new file."
        if inp.old_string == inp.new_string:
            return "old_string and new_string are identical; nothing to change."
        if inp.old_string == "":
            return "old_string is empty. Use write_file to replace a whole file."
        if error := _stale_read_error(p, ctx):
            return error
        text = _read_text(p)
        old, _ = _match_line_endings(text, inp.old_string, inp.new_string)
        count = text.count(old)
        if count == 0:
            return (
                f"old_string was not found in {inp.path}. It must match exactly, including "
                "whitespace and indentation. Read the file again and copy the text precisely."
            )
        if count > 1 and not inp.replace_all:
            return (
                f"old_string occurs {count} times in {inp.path}. Include more surrounding lines to "
                "make it unique, or set replace_all to true."
            )
        return None

    async def check_permissions(self, inp: EditFileInput, ctx: ToolContext) -> PermissionDecision:
        p = ctx.resolve(inp.path)
        if not ctx.in_workspace(p):
            return PermissionDecision.ask(f"editing {p}, which is outside the workspace")
        return PermissionDecision.passthrough()

    async def call(self, inp: EditFileInput, ctx: ToolContext) -> ToolResult:
        p = ctx.resolve(inp.path)
        if error := _stale_read_error(p, ctx):  # re-check: time passed while waiting for approval
            raise ToolError(error)
        text = _read_text(p)
        old, new = _match_line_endings(text, inp.old_string, inp.new_string)
        count = text.count(old)
        if count == 0 or (count > 1 and not inp.replace_all):
            raise ToolError(f"{inp.path} changed; old_string no longer matches as expected. Read it again.")
        first = text.index(old)
        updated = text.replace(old, new) if inp.replace_all else text.replace(old, new, 1)
        _write_text(p, updated)
        ctx.read_files[p] = p.stat().st_mtime_ns

        lines = updated.splitlines()
        start_line = updated[:first].count("\n")
        lo = max(0, start_line - 3)
        hi = min(len(lines), start_line + new.count("\n") + 4)
        snippet = _numbered(lines[lo:hi], lo + 1)
        n = count if inp.replace_all else 1
        return ToolResult(f"Edited {ctx.display(p)} ({n} replacement{'s' if n != 1 else ''}). Result:\n{snippet}")


def _match_line_endings(text: str, old: str, new: str) -> tuple[str, str]:
    """Models write '\\n'; make that match files that use '\\r\\n'."""
    if "\r\n" in text and "\r\n" not in old and old not in text:
        return old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
    return old, new
