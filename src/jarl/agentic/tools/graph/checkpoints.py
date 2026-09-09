"""Read checkpoint overview or filtered candidates for a node.

Use this for checkpoint selection before fork/extend. Prefer overview first;
add filters only when you need a sorted or ranged page of candidates.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.handler import wire_handler
from jarl.agentic.tools.specs import ToolSpec
from jarl.operations.contracts.constants import CHECKPOINTS_DEFAULT_LIMIT, CHECKPOINTS_MAX_LIMIT
from jarl.operations.graph.checkpoints import CheckpointsRequest, checkpoints

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_graph_checkpoints"]


class ToolRequest(BaseModel):
    """Inputs for reading node checkpoints."""

    node_id: str | None = Field(
        default=None,
        description="Node whose checkpoints to inspect. Omit for the current node.",
    )
    step_min: int | None = Field(
        default=None,
        description="Inclusive lower bound on checkpoint_step. Triggers search mode.",
    )
    step_max: int | None = Field(
        default=None,
        description="Inclusive upper bound on checkpoint_step. Triggers search mode.",
    )
    sort_by: str | None = Field(
        default=None,
        description=(
            "Metric key or 'checkpoint_step' used to order candidates. "
            "Typical eval keys: eval/episode_return, eval/episode_length. "
            "Triggers search mode."
        ),
    )
    sort_descending: bool = Field(
        default=True,
        description="When sorting candidates, higher values first. Default true.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=CHECKPOINTS_MAX_LIMIT,
        description=(
            f"Max candidates to return in search mode (default {CHECKPOINTS_DEFAULT_LIMIT}, "
            f"cap {CHECKPOINTS_MAX_LIMIT}). Setting limit triggers search mode."
        ),
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Skip this many ordered candidates before the page. Triggers search mode when > 0.",
    )


TOOL_SPEC = ToolSpec(
    name="graph_checkpoints",
    description=(
        "Read a node's checkpoint overview (latest/final/best) or a filtered "
        "candidate page by step range, metric sort, and limit. Use before "
        "fork/extend to choose checkpoint_step. Not for recovery diagnosis or "
        "metric curve interpretation. Does not mutate the graph."
    ),
    labels=frozenset({"graph", "read", "operate"}),
)


def run_graph_checkpoints(ctx: ToolContext, request: ToolRequest) -> str:
    """Read checkpoints via ``jarl.operations.graph.checkpoints``."""
    response = checkpoints(
        ctx.graph,
        CheckpointsRequest(
            node_id=request.node_id,
            step_min=request.step_min,
            step_max=request.step_max,
            sort_by=request.sort_by,
            sort_descending=request.sort_descending,
            limit=request.limit,
            offset=request.offset,
        ),
    )
    return json.dumps(response.to_compact_dict(), ensure_ascii=False, default=str)


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(workflow, run_graph_checkpoints, TOOL_SPEC, request_cls=ToolRequest)
