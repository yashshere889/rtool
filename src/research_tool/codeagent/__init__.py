"""A small, model-agnostic tool system for coding agents.

Vendored from github.com/yashshere889/coding-agent (commit 5c04c9e) so any rtool agent can run a
tool-calling loop; ``RToolModel`` replaces the original OpenAI SDK adapter. rtool's entry point for
agents is ``research_tool.agents.tool_agent.run_tool_agent``.
"""

from .executor import ToolCall, ToolExecutor, ToolOutcome
from .hooks import Hooks, PreToolUseOutcome
from .llm import AssistantTurn, ChatModel, RToolModel
from .loop import DEFAULT_SYSTEM_PROMPT, Agent
from .permissions import ApprovalRequest, ApprovalResponse, PermissionEngine, Rule
from .registry import ToolRegistry
from .tool import CancelToken, PermissionDecision, Tool, ToolContext, ToolError, ToolInput, ToolResult
from .tools import default_tools

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "Agent",
    "ApprovalRequest",
    "ApprovalResponse",
    "AssistantTurn",
    "CancelToken",
    "ChatModel",
    "Hooks",
    "PermissionDecision",
    "PermissionEngine",
    "PreToolUseOutcome",
    "RToolModel",
    "Rule",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolError",
    "ToolExecutor",
    "ToolInput",
    "ToolOutcome",
    "ToolRegistry",
    "ToolResult",
    "default_tools",
]
