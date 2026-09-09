"""Append a child on the same branch from that branch's head.

Linear continuation on an existing branch line, even when current_node is older
on that branch. Does not run training.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.handler import wire_handler
from jarl.agentic.tools.specs import ToolSpec
from jarl.experiments.io.checkpoints import CheckpointRef
from jarl.operations.graph.extend import ExtendRequest, extend
from jarl.training.override_models import ForkConfigOverrides

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_graph_extend"]


class CheckpointRefRequest(BaseModel):
    """Checkpoint reference for warm-starting the child."""

    node_id: str = Field(
        description="Node that owns the checkpoint to restore, typically the branch head.",
    )
    checkpoint_step: int = Field(description="Checkpoint step within that node to restore.")


class ToolRequest(BaseModel):
    """Inputs for extending a branch from its head."""

    label: str = Field(
        default="",
        description="Human-readable label for the new child. Omit for an auto label.",
    )
    branch: str | None = Field(
        default=None,
        description="Branch to extend. Defaults to the current node's branch.",
    )
    config_overrides: ForkConfigOverrides | None = Field(
        default=None,
        description=(
            "Configuration changes for the child using dotted schema keys "
            "(not config_highlights names). Omitted values inherit from the branch head."
        ),
    )
    from_checkpoint: CheckpointRefRequest | None = Field(
        default=None,
        description="Warm-start the child from a saved checkpoint.",
    )
    prepare: bool = Field(
        default=False,
        description="Create the child in prepared status without opening training writers.",
    )


TOOL_SPEC = ToolSpec(
    name="graph_extend",
    description=(
        "Append a child on the same branch from that branch's head—the next linear "
        "step, even when current_node is older on that branch. Does not run training; "
        "leaves a prepared or created child as current on that branch. Not for a new "
        "branch name or a parent that is not that branch's head."
    ),
    labels=frozenset({"graph", "mutation", "operate"}),
)


def run_graph_extend(ctx: ToolContext, request: ToolRequest) -> str:
    """Extend a branch via ``jarl.operations.graph.extend``."""
    response = extend(
        ctx.graph,
        ExtendRequest(
            label=request.label,
            branch=request.branch,
            config_overrides=(
                request.config_overrides.to_dotted_dict() if request.config_overrides is not None else None
            ),
            from_checkpoint=_checkpoint_ref(request.from_checkpoint),
            prepare=request.prepare,
        ),
    )
    ctx.replace_graph(ctx.graph)
    return json.dumps(response.to_compact_dict(), ensure_ascii=False, default=str)


def _checkpoint_ref(value: CheckpointRefRequest | None) -> CheckpointRef | None:
    if value is None:
        return None
    return CheckpointRef(node_id=value.node_id, checkpoint_step=value.checkpoint_step)


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(workflow, run_graph_extend, TOOL_SPEC, request_cls=ToolRequest)
