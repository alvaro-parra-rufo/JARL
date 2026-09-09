"""Fork a new branch from a chosen parent node.

Tree branching: a new branch name or a parent that is not the branch head of an
existing line. Does not run training.
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
from jarl.operations.graph.fork import ForkRequest, fork
from jarl.training.override_models import ForkConfigOverrides

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_graph_fork"]


class CheckpointRefRequest(BaseModel):
    """Checkpoint reference for warm-starting the child."""

    node_id: str = Field(
        description="Node that owns the checkpoint to restore, typically the parent node.",
    )
    checkpoint_step: int = Field(description="Checkpoint step within that node to restore.")


class ToolRequest(BaseModel):
    """Inputs for forking a branch."""

    branch: str = Field(description="Name for the new branch.")
    label: str = Field(
        default="",
        description="Human-readable label for the new child. Omit for an auto label.",
    )
    from_node: str | None = Field(
        default=None,
        description="Parent to fork from. Defaults to the current node.",
    )
    config_overrides: ForkConfigOverrides | None = Field(
        default=None,
        description=(
            "Configuration changes for the child using dotted schema keys "
            "(not config_highlights names). Omitted values inherit from the parent."
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
    name="graph_fork",
    description=(
        "Start a new branch from a chosen parent—a new branch name or a parent that is "
        "not the branch head of an existing line. Does not run training or append on the "
        "same branch from its head; leaves a prepared or created child as current on "
        "the new branch."
    ),
    labels=frozenset({"graph", "mutation", "operate"}),
)


def run_graph_fork(ctx: ToolContext, request: ToolRequest) -> str:
    """Fork a branch via ``jarl.operations.graph.fork``."""
    response = fork(
        ctx.graph,
        ForkRequest(
            branch=request.branch,
            label=request.label,
            from_node=request.from_node,
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
    return wire_handler(workflow, run_graph_fork, TOOL_SPEC, request_cls=ToolRequest)
