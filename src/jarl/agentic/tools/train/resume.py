"""Resume training on a failed or interrupted node."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.handler import wire_handler
from jarl.agentic.tools.specs import ToolSpec
from jarl.agentic.tools.train._form import (
    RunFormPayloadRequest,
    build_run_form_payload,
    resolve_trainer,
)
from jarl.operations.train.resume import ResumeRequest, resume
from jarl.training.launch import train_detach_enabled
from jarl.training.override_models import RetrainConfigOverrides

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_train_resume"]


class ToolRequest(BaseModel):
    """Inputs for resuming training on a node."""

    node_id: str | None = Field(
        default=None,
        description="Node with a failed or interrupted run to resume. Defaults to the current node.",
    )
    config_overrides: RetrainConfigOverrides | None = Field(
        default=None,
        description=("Configuration changes for the resumed run. Omitted values inherit from the node."),
    )
    form: RunFormPayloadRequest | None = Field(
        default=None,
        description="Preset-style training parameters instead of `config_overrides`.",
    )


TOOL_SPEC = ToolSpec(
    name="train_resume",
    description=(
        "Resume training on a failed or interrupted node from its last checkpoint. "
        "Over MCP this launches a subprocess and returns immediately with pid/log_path; "
        "poll session_status until the node is completed or failed. Does not add a "
        "graph node. Not for first training on a prepared node or completed runs."
    ),
    labels=frozenset({"train", "operate"}),
)


def run_train_resume(ctx: ToolContext, request: ToolRequest) -> str:
    """Resume training via ``jarl.operations.train.resume``."""
    detach = train_detach_enabled()
    trainer_node_id = request.node_id or ctx.try_current_node_id()
    response = resume(
        ctx.graph,
        ResumeRequest(
            node_id=request.node_id,
            payload=build_run_form_payload(request.form),
            config_overrides=(
                request.config_overrides.to_dotted_dict() if request.config_overrides is not None else None
            ),
            detach=detach,
        ),
        trainer=None if detach else resolve_trainer(ctx, node_id=trainer_node_id),
    )
    ctx.replace_graph(ctx.graph)
    return json.dumps(response.to_compact_dict(), ensure_ascii=False, default=str)


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(workflow, run_train_resume, TOOL_SPEC, request_cls=ToolRequest)
