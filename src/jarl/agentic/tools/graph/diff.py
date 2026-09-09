"""Compare resolved configs between two experiment nodes."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.handler import wire_handler
from jarl.agentic.tools.specs import ToolSpec
from jarl.operations.graph.diff import DiffRequest, diff

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_graph_diff"]


class ToolRequest(BaseModel):
    """Nodes to compare."""

    node_a: str = Field(
        description="First node to compare. Use ids from session_status or graph_summary.",
    )
    node_b: str = Field(
        description="Second node to compare. Use ids from session_status or graph_summary.",
    )


TOOL_SPEC = ToolSpec(
    name="graph_diff",
    description=("Read config differences between two nodes. Does not mutate the graph."),
    labels=frozenset({"graph", "read", "operate"}),
)


def run_graph_diff(ctx: ToolContext, request: ToolRequest) -> str:
    """Return a config diff via ``jarl.operations.graph.diff``."""
    response = diff(
        ctx.graph,
        DiffRequest(node_a=request.node_a, node_b=request.node_b),
    )
    return json.dumps(response.to_compact_dict(), ensure_ascii=False, default=str)


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(workflow, run_graph_diff, TOOL_SPEC, request_cls=ToolRequest)
