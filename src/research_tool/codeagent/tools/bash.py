"""Shell tool with a conservative command analyser for permissions.

The analyser splits a command into simple commands at ``&&``, ``||``, ``;``, ``|``, ``&``,
parentheses and newlines, then:

* treats the call as read-only (auto-allowed, parallel-safe) only if every simple command is on
  a small read-only allowlist and nothing redirects output to a file;
* gives permission rules one subject per simple command, so ``bash(git *)`` cannot be abused
  with ``git status && rm -rf ~``;
* refuses to auto-allow anything it cannot analyse (``$(...)``, backticks, process
  substitution, unbalanced quotes): those always go to a person.

This is a heuristic, not a sandbox. For untrusted tasks, run the whole agent inside a
container or VM and use the permission layer as a second line of defence.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import signal

from pydantic import Field

from ..tool import PermissionDecision, Tool, ToolContext, ToolError, ToolInput, ToolResult

READ_ONLY_COMMANDS = {
    "ls", "cat", "head", "tail", "wc", "pwd", "echo", "which", "file", "stat", "tree",
    "rg", "grep", "du", "df", "date", "uname", "whoami", "diff", "cut", "basename",
    "dirname", "realpath", "true", "find", "git",
}  # fmt: skip
GIT_READ_ONLY = {"status", "log", "diff", "show", "rev-parse", "ls-files", "blame", "grep", "shortlog", "describe"}
UNSAFE_FLAGS = {
    "find": ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"),
    "rg": ("--pre",),
    "tree": ("-o",),
    "git": ("--output", "-O", "--open-files-in-pager", "--ext-diff"),
}
_UNANALYSABLE = ("$(", "`", "<(", ">(")


def split_command(command: str) -> list[list[str]] | None:
    """Split into simple commands (lists of words). None if the command cannot be analysed safely.

    Redirection operators stay in the word lists so callers can see them.
    """
    if any(s in command for s in _UNANALYSABLE):
        return None
    lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:  # unbalanced quotes
        return None
    commands: list[list[str]] = [[]]
    for tok in tokens:
        if tok and set(tok) <= set(";&|()"):
            if commands[-1]:
                commands.append([])
        else:
            commands[-1].append(tok)
    return [c for c in commands if c]


def _writes_output(words: list[str]) -> bool:
    for i, w in enumerate(words):
        if set(w) <= set("<>&") and ">" in w:
            target = words[i + 1] if i + 1 < len(words) else ""
            if w.endswith("&") and target.isdigit():  # 2>&1 duplicates a descriptor
                continue
            if target == "/dev/null":
                continue
            return True
    return False


def is_read_only_command(command: str) -> bool:
    parts = split_command(command)
    if not parts:
        return False
    for words in parts:
        if _writes_output(words):
            return False
        prog = os.path.basename(words[0])
        if prog not in READ_ONLY_COMMANDS:
            return False
        if any(w.startswith(flag) for w in words[1:] for flag in UNSAFE_FLAGS.get(prog, ())):
            return False
        if prog == "git" and (len(words) < 2 or words[1] not in GIT_READ_ONLY):
            return False
    return True


def _subject(words: list[str]) -> str:
    return " ".join(w for w in words if not (set(w) <= set("<>&")))


class BashInput(ToolInput):
    command: str = Field(description="The shell command to run with bash")
    timeout_s: float = Field(120, gt=0, le=600, description="Kill the command after this many seconds")
    description: str | None = Field(None, description="A few words saying what the command does, shown to the user")


class BashTool(Tool[BashInput]):
    name = "bash"
    description = (
        "Run a shell command in the working directory and return its combined stdout/stderr and exit "
        "code. Use it for builds, tests, git and package managers. Use read_file, glob and grep instead "
        "of cat, find and grep for reading and searching. Commands are non-interactive: no stdin."
    )
    Input = BashInput
    max_result_chars = 30_000

    def is_read_only(self, inp: BashInput) -> bool:
        return is_read_only_command(inp.command)

    def is_concurrency_safe(self, inp: BashInput) -> bool:
        return self.is_read_only(inp)

    def permission_subjects(self, inp: BashInput) -> list[str]:
        parts = split_command(inp.command)
        return [_subject(w) for w in parts] if parts else [inp.command]

    def describe_call(self, inp: BashInput) -> str:
        return f"bash: {inp.command}" + (f"  # {inp.description}" if inp.description else "")

    async def check_permissions(self, inp: BashInput, ctx: ToolContext) -> PermissionDecision:
        if split_command(inp.command) is None:
            return PermissionDecision.ask(
                "the command uses substitution or could not be parsed, so rules cannot vouch for it"
            )
        return PermissionDecision.passthrough()

    async def call(self, inp: BashInput, ctx: ToolContext) -> ToolResult:
        proc = await asyncio.create_subprocess_exec(
            "/bin/bash", "-c", inp.command,
            cwd=ctx.cwd,
            env=ctx.env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,  # own process group, so we can kill children too
        )  # fmt: skip
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=inp.timeout_s)
        except TimeoutError:
            _kill_group(proc)
            await proc.wait()
            raise ToolError(f"Command timed out after {inp.timeout_s:g}s and was killed.") from None
        except asyncio.CancelledError:
            _kill_group(proc)
            raise
        text = out.decode("utf-8", errors="replace").rstrip()
        rc = proc.returncode
        body = text or "(no output)"
        if rc != 0:
            body += f"\n[exit code {rc}]"
        return ToolResult(body, metadata={"exit_code": rc})


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
