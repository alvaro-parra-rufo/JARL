"""Patch the configurable reward weights on a node that has not trained yet.

Omitted channels keep their current weights. Native reward starts from the task
default mix. Does not fork, extend, or run training.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.handler import wire_handler
from jarl.agentic.tools.specs import ToolSpec
from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.operations.graph.set_reward import SetRewardRequest, set_reward

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_graph_set_reward"]

_WEIGHT_FIELDS = {
    name: (
        float | None,
        Field(
            default=None,
            ge=0.0,
            allow_inf_nan=False,
            description=info.description,
        ),
    )
    for name, info in RewardWeightsConfig.model_fields.items()
}

ToolRequest = cast(
    "type[BaseModel]",
    create_model(
        "ToolRequest",
        __config__=ConfigDict(extra="forbid"),
        node_id=(
            str | None,
            Field(
                default=None,
                description="Node to update. Defaults to the current node, which must not have trained yet.",
            ),
        ),
        **_WEIGHT_FIELDS,
    ),
)
"""Sparse patch of configurable reward weights."""


TOOL_SPEC = ToolSpec(
    name="graph_set_reward",
    description=(
        "Patch the configurable reward weights of a node that has not trained yet. "
        "Useful when changing the reward configuration is relevant to the current "
        "task or to addressing reward-driven behavior. Omitted channels keep their "
        "current weights. Native reward starts from the task default mix. "
        "Does not fork, extend, or run training."
    ),
    labels=frozenset({"graph", "mutation", "operate"}),
)


def run_graph_set_reward(ctx: ToolContext, request: BaseModel) -> str:
    """Patch the configurable reward mix via ``jarl.operations.graph.set_reward``."""
    response = set_reward(
        ctx.graph,
        SetRewardRequest(node_id=_optional_node_id(request), weights=_weight_patch(request)),
    )
    ctx.replace_graph(ctx.graph)
    return json.dumps(response.to_compact_dict(), ensure_ascii=False, default=str)


def _optional_node_id(request: BaseModel) -> str | None:
    node_id = request.model_dump(exclude_unset=True).get("node_id")
    return str(node_id) if node_id is not None else None


def _weight_patch(request: BaseModel) -> dict[str, float]:
    payload = request.model_dump(exclude={"node_id"}, exclude_unset=True, exclude_none=True)
    return {name: float(value) for name, value in payload.items()}


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(workflow, run_graph_set_reward, TOOL_SPEC, request_cls=ToolRequest)
