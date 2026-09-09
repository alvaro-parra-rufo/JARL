"""Read the configurable reward mix on a node.

Returns null when the environment uses its native reward. Does not mutate the
graph. Not for changing weights.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.handler import wire_handler
from jarl.agentic.tools.specs import ToolSpec
from jarl.operations.graph.reward import RewardRequest, reward

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_graph_reward"]


class ToolRequest(BaseModel):
    """Inputs for reading a node's configurable reward mix."""

    model_config = ConfigDict(extra="forbid")

    node_id: str | None = Field(
        default=None,
        description="Node whose mix to read. Defaults to the current node.",
    )


TOOL_SPEC = ToolSpec(
    name="graph_reward",
    description=(
        "Read the configurable reward weights of a node. Use when reward configuration "
        "is relevant to the current task, such as inspecting or diagnosing reward-driven "
        "behavior. Returns null when the environment uses its native reward. "
        "Read-only; does not mutate the graph. Use graph_set_reward to change weights."
    ),
    labels=frozenset({"graph", "read", "operate"}),
)


def run_graph_reward(ctx: ToolContext, request: ToolRequest) -> str:
    """Read the configurable reward mix via ``jarl.operations.graph.reward``."""
    response = reward(ctx.graph, RewardRequest(node_id=request.node_id))
    return json.dumps(response.to_compact_dict(), ensure_ascii=False, default=str)


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(workflow, run_graph_reward, TOOL_SPEC, request_cls=ToolRequest)
