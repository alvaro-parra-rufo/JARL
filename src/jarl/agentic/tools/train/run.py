"""Run training on a prepared node or create a root in an empty experiment."""

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
from jarl.operations.train.run import RunRequest, run
from jarl.training.launch import train_detach_enabled
from jarl.training.override_models import RetrainConfigOverrides

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_train_run"]


class ToolRequest(BaseModel):
    """Inputs for running training on a node."""

    node_id: str | None = Field(
        default=None,
        description="Prepared node to train. Defaults to the current node.",
    )
    create_root: bool = Field(
        default=False,
        description=("When the experiment has no nodes, create a root and train. Ignores `node_id`."),
    )
    branch: str = Field(
        default="main",
        description="Branch for the new root when `create_root` is true.",
    )
    label: str = Field(
        default="",
        description="Label for the new root when `create_root` is true. Omit for an auto label.",
    )
    config_overrides: RetrainConfigOverrides | None = Field(
        default=None,
        description=("Configuration changes for the run. Omitted values inherit from the node."),
    )
    form: RunFormPayloadRequest | None = Field(
        default=None,
        description="Preset-style training parameters instead of `config_overrides`.",
    )


TOOL_SPEC = ToolSpec(
    name="train_run",
    description=(
        "Start training on a prepared node, or create and train a root in an empty "
        "experiment. Over MCP this launches a subprocess and returns immediately "
        "with pid/log_path; poll session_status until the node is completed or "
        "failed. Does not add a graph node or change the map. Not for resuming "
        "a failed or interrupted run."
    ),
    labels=frozenset({"train", "operate"}),
)


def run_train_run(ctx: ToolContext, request: ToolRequest) -> str:
    """Run training via ``jarl.operations.train.run``."""
    detach = train_detach_enabled()
    trainer_node_id = None if request.create_root else (request.node_id or ctx.try_current_node_id())
    response = run(
        ctx.graph,
        RunRequest(
            payload=build_run_form_payload(request.form),
            node_id=request.node_id,
            config_overrides=(
                request.config_overrides.to_dotted_dict() if request.config_overrides is not None else None
            ),
            create_root=request.create_root,
            branch=request.branch,
            label=request.label,
            detach=detach,
        ),
        trainer=None if detach else resolve_trainer(ctx, node_id=trainer_node_id),
    )
    ctx.replace_graph(ctx.graph)
    return json.dumps(response.to_compact_dict(), ensure_ascii=False, default=str)


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(workflow, run_train_run, TOOL_SPEC, request_cls=ToolRequest)
