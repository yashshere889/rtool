from ..tool import Tool
from .bash import BashTool
from .files import EditFileTool, ReadFileTool, WriteFileTool
from .search import GlobTool, GrepTool


def default_tools() -> list[Tool]:
    return [ReadFileTool(), EditFileTool(), WriteFileTool(), GlobTool(), GrepTool(), BashTool()]


__all__ = ["BashTool", "EditFileTool", "GlobTool", "GrepTool", "ReadFileTool", "WriteFileTool", "default_tools"]
