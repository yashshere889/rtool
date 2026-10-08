"""Everything needed to write an agent, in one import::

from research_tool.agents.base import Agent, AgentContext, StageFailed, artifact_type, register_agent
"""

from ..core.agent import Agent, Task
from ..core.artifacts import artifact_type
from ..core.context import AgentContext
from ..core.errors import StageFailed
from ..core.registry import register_agent
from ..core.state import FeedbackItem, FeedbackKind, GateResolution

__all__ = [
    "Agent",
    "AgentContext",
    "FeedbackItem",
    "FeedbackKind",
    "GateResolution",
    "StageFailed",
    "Task",
    "artifact_type",
    "register_agent",
]
