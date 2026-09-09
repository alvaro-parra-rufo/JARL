"""Read structured train recovery status for a failed or interrupted node."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.handler import wire_handler
from jarl.agentic.tools.specs import ToolSpec
from jarl.operations.train.recovery_status import RecoveryStatusRequest, recovery_status

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_train_recovery_status"]


class ToolRequest(BaseModel):
    """Inputs for reading train recovery status."""

    node_id: str | None = Field(
        default=None,
        description="Node to diagnose for resume eligibility. Omit for the current node.",
    )


TOOL_SPEC = ToolSpec(
    name="train_recovery_status",
    description=(
        "Diagnose whether a failed or interrupted node can resume training. "
        "Returns facts plus recommended_action train_resume or none. "
        "Not for checkpoint ranking or metric curves. Does not mutate the graph "
        "and does not resume training."
    ),
    labels=frozenset({"train", "read", "operate"}),
)


def run_train_recovery_status(ctx: ToolContext, request: ToolRequest) -> str:
    """Read recovery status via ``jarl.operations.train.recovery_status``."""
    response = recovery_status(
        ctx.graph,
        RecoveryStatusRequest(node_id=request.node_id),
    )
    return json.dumps(response.to_compact_dict(), ensure_ascii=False, default=str)


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(
        workflow,
        run_train_recovery_status,
        TOOL_SPEC,
        request_cls=ToolRequest,
    )
