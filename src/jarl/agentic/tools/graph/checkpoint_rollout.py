"""Execute and summarize one reproducible rollout from an exact checkpoint."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.handler import wire_handler
from jarl.agentic.tools.specs import ToolSpec
from jarl.operations.contracts.constants import (
    CHECKPOINT_ROLLOUT_MAX_STEPS,
    CHECKPOINT_ROLLOUT_RECORD_VIDEO_DEFAULT,
    CHECKPOINT_ROLLOUT_VIDEO_VIEW_MODE_DEFAULT,
)
from jarl.operations.graph.checkpoint_rollout import (
    CheckpointRolloutRequest,
    checkpoint_rollout,
)

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = [
    "TOOL_SPEC",
    "ToolRequest",
    "build_handler",
    "run_graph_checkpoint_rollout",
]


class ToolRequest(BaseModel):
    """Inputs for executing one checkpoint rollout."""

    checkpoint_step: int = Field(
        ge=0,
        description="Exact checkpoint step to restore. Use graph_checkpoints when the step is unknown.",
    )
    seed: int = Field(
        description="Environment seed for this reproducible episode.",
    )
    node_id: str | None = Field(
        default=None,
        description="Node that owns the checkpoint. Omit for the current node.",
    )
    env_id: str | None = Field(
        default=None,
        description=(
            "Target Navix environment. Omit to use the node environment; "
            "checkpoint transfer compatibility is validated."
        ),
    )
    max_steps: int | None = Field(
        default=None,
        ge=1,
        le=CHECKPOINT_ROLLOUT_MAX_STEPS,
        description="Episode horizon. Omit to use the target environment horizon.",
    )
    record_video: bool = Field(
        default=CHECKPOINT_ROLLOUT_RECORD_VIDEO_DEFAULT,
        description="Whether to persist an MP4 captured during the same rollout.",
    )
    video_view_mode: Literal["full", "first_person"] = Field(
        default=CHECKPOINT_ROLLOUT_VIDEO_VIEW_MODE_DEFAULT,
        description="Camera used when video recording is requested.",
    )


TOOL_SPEC = ToolSpec(
    name="graph_checkpoint_rollout",
    description=(
        "Execute one reproducible greedy rollout from an exact checkpoint step "
        "and return a bounded analysis with persisted artifact paths. The payload "
        "includes `checkpoint_eval.episode_return` and `checkpoint_eval.episode_length` "
        "from training when available. Not for checkpoint selection or ranking "
        "(graph_checkpoints). Does not train or mutate graph topology."
    ),
    labels=frozenset({"graph", "read", "inference", "operate"}),
)


def run_graph_checkpoint_rollout(
    ctx: ToolContext,
    request: ToolRequest,
) -> str:
    """Execute a rollout via ``jarl.operations.graph.checkpoint_rollout``."""
    response = checkpoint_rollout(
        ctx.graph,
        CheckpointRolloutRequest(
            checkpoint_step=request.checkpoint_step,
            seed=request.seed,
            node_id=request.node_id,
            env_id=request.env_id,
            max_steps=request.max_steps,
            record_video=request.record_video,
            video_view_mode=request.video_view_mode,
        ),
    )
    return json.dumps(
        response.to_compact_dict(),
        ensure_ascii=False,
        default=str,
    )


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(
        workflow,
        run_graph_checkpoint_rollout,
        TOOL_SPEC,
        request_cls=ToolRequest,
    )
