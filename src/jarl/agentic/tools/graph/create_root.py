"""Create the first prepared root node in an empty experiment."""

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
    RunFormValuesRequest,
    build_run_form_payload,
)
from jarl.operations.graph.create_root import CreateRootRequest, create_root
from jarl.training.presets import AlgorithmChoice, PresetKind, RunFormPayload

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_graph_create_root"]


class ToolRequest(BaseModel):
    """Inputs for creating a prepared root node."""

    label: str = Field(description="Human-readable label for the root.")
    branch: str = Field(default="main", description="Branch name for the root.")
    preset: PresetKind = Field(
        default="fast",
        description="Training preset for the root config.",
    )
    algorithm: AlgorithmChoice = Field(
        default="ppo",
        description="Algorithm when `preset` is `fast`.",
    )
    form: RunFormPayloadRequest | None = Field(
        default=None,
        description=(
            "Training form fields for the root config. Use with `preset: custom` "
            "or to override fields such as `max_episode_steps`."
        ),
    )


TOOL_SPEC = ToolSpec(
    name="graph_create_root",
    description=(
        "Create a prepared root in an empty experiment. Does not run training. "
        "Not for experiments that already have nodes."
    ),
    labels=frozenset({"graph", "initialization", "setup"}),
)


def run_graph_create_root(ctx: ToolContext, request: ToolRequest) -> str:
    """Create a root node via ``jarl.operations.graph.create_root``."""
    response = create_root(
        ctx.graph,
        CreateRootRequest(
            label=request.label,
            branch=request.branch,
            preset=request.preset,
            algorithm=request.algorithm,
            form=_resolve_create_root_form(request),
        ),
    )
    ctx.replace_graph(ctx.graph)
    return json.dumps(response.to_compact_dict(), ensure_ascii=False, default=str)


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(workflow, run_graph_create_root, TOOL_SPEC, request_cls=ToolRequest)


def _resolve_create_root_form(request: ToolRequest) -> RunFormPayload | None:
    """Build a root form payload from the tool request, if needed."""
    if request.form is None and request.preset == "fast":
        return None
    if request.form is None:
        form_request = RunFormPayloadRequest(
            values=RunFormValuesRequest(
                preset=request.preset,
                algorithm=request.algorithm,
            ),
        )
    else:
        base_values = request.form.values or RunFormValuesRequest()
        merged_values = base_values.model_copy(
            update={
                key: value
                for key, value in {
                    "preset": request.preset,
                    "algorithm": request.algorithm,
                }.items()
                if getattr(base_values, key) is None
            }
        )
        form_request = request.form.model_copy(update={"values": merged_values})
    return build_run_form_payload(form_request)
