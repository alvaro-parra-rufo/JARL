"""Return a depth-limited visible subtree for tree exploration."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.handler import wire_handler
from jarl.agentic.tools.specs import ToolSpec
from jarl.operations.contracts.constants import SUBTREE_DEFAULT_DEPTH
from jarl.operations.graph.subtree import SubtreeRequest, subtree

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_graph_subtree"]


class ToolRequest(BaseModel):
    """Inputs for building a visible subtree."""

    root_id: str | None = Field(
        default=None,
        description="Subtree root. Defaults to the experiment tree root.",
    )
    depth: int = Field(
        default=SUBTREE_DEFAULT_DEPTH,
        description="Levels below the root to include.",
    )
    metric_key: str = Field(
        default="rollout/episode_return",
        description="Metric shown on each node in the snapshot.",
    )


TOOL_SPEC = ToolSpec(
    name="graph_subtree",
    description=("Read a depth-limited subtree with rollout metrics on each node. Does not mutate the graph."),
    labels=frozenset({"graph", "read", "operate"}),
)


def run_graph_subtree(ctx: ToolContext, request: ToolRequest) -> str:
    """Build a visible subtree via ``jarl.operations.graph.subtree``."""
    response = subtree(
        ctx.graph,
        SubtreeRequest(
            root_id=request.root_id,
            depth=request.depth,
            metric_key=request.metric_key,
        ),
    )
    return json.dumps(response.to_compact_dict(), ensure_ascii=False, default=str)


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(workflow, run_graph_subtree, TOOL_SPEC, request_cls=ToolRequest)
