"""Checkout an experiment node without changing branch heads."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.handler import wire_handler
from jarl.agentic.tools.specs import ToolSpec
from jarl.operations.graph.checkout import CheckoutRequest, checkout

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_graph_checkout"]


class ToolRequest(BaseModel):
    """Target node for checkout."""

    node_id: str = Field(
        description="Node to select as current. Use ids from session_status or graph_summary.",
    )


TOOL_SPEC = ToolSpec(
    name="graph_checkout",
    description=("Set the current node without changing the graph. Does not fork, extend, or train."),
    labels=frozenset({"graph", "mutation", "operate"}),
)


def run_graph_checkout(ctx: ToolContext, request: ToolRequest) -> str:
    """Checkout a node via ``jarl.operations.graph.checkout``."""
    response = checkout(ctx.graph, CheckoutRequest(node_id=request.node_id))
    ctx.replace_graph(ctx.graph)
    return json.dumps(response.to_compact_dict(), ensure_ascii=False, default=str)


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(workflow, run_graph_checkout, TOOL_SPEC, request_cls=ToolRequest)
