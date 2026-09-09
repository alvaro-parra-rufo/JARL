"""Return an enriched experiment or node summary."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.handler import wire_handler
from jarl.agentic.tools.specs import ToolSpec
from jarl.operations.graph.summary import SummaryRequest, summary

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_graph_summary"]


class ToolRequest(BaseModel):
    """Inputs for building an experiment summary."""

    node_id: str | None = Field(
        default=None,
        description="Node to summarize. Omit for the whole experiment.",
    )


TOOL_SPEC = ToolSpec(
    name="graph_summary",
    description=(
        "Read experiment or node summary with config highlights and status. "
        "Not for checkpoint aliases, recovery diagnosis, or metric curves "
        "(use graph_checkpoints or train_recovery_status). Does not mutate the graph."
    ),
    labels=frozenset({"graph", "read", "operate"}),
)


def run_graph_summary(ctx: ToolContext, request: ToolRequest) -> str:
    """Build an enriched summary via ``jarl.operations.graph.summary``."""
    response = summary(ctx.graph, SummaryRequest(node_id=request.node_id))
    return json.dumps(response.to_compact_dict(), ensure_ascii=False, default=str)


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(workflow, run_graph_summary, TOOL_SPEC, request_cls=ToolRequest)
