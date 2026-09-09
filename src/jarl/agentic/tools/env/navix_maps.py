"""List Navix maps by category."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.handler import wire_handler
from jarl.agentic.tools.specs import ToolSpec
from jarl.envs.navix.maps import navix_map_category_slugs
from jarl.operations.env.navix_maps import NavixMapsRequest, navix_maps
from jarl.utils.pydantic import FlexibleList

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_env_navix_maps"]

_CATEGORY_KEYS = ", ".join(navix_map_category_slugs())
"""Exact category keys accepted by ``env_navix_maps``."""

_CATEGORIES_EXAMPLE = '["door_key"]'
"""Canonical single-key example payload for the ``categories`` list field."""

_CATEGORIES_MULTI_EXAMPLE = '["door_key", "key_corridor"]'
"""Canonical multi-key example payload for the ``categories`` list field."""


class ToolRequest(BaseModel):
    """Inputs for listing Navix maps."""

    categories: FlexibleList[str] | None = Field(
        default=None,
        description=(
            "Exact map-family keys (OR). JSON array or comma-separated string. "
            f"Examples: {_CATEGORIES_EXAMPLE} or {_CATEGORIES_MULTI_EXAMPLE}. "
            f"Allowed keys: {_CATEGORY_KEYS}."
        ),
    )
    query: str | None = Field(
        default=None,
        description=(
            "Keyword search over env id, description, category label and aliases "
            "(e.g. 'llave', 'lava', 'random'). All whitespace-separated tokens must match."
        ),
    )
    difficulty_min: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Inclusive lower bound on heuristic difficulty in [1, 100].",
    )
    difficulty_max: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Inclusive upper bound on heuristic difficulty in [1, 100].",
    )
    jarl_transfer_ready_only: bool = Field(
        default=False,
        description="When true, keep only maps registered for JARL checkpoint transfer.",
    )
    installed_only: bool = Field(
        default=False,
        description="When true, keep only maps available in the local Navix package.",
    )


TOOL_SPEC = ToolSpec(
    name="env_navix_maps",
    description=(
        "Read Navix maps by categories, keywords and difficulty with install/transfer "
        "status.Pass categories as a JSON array of exact keys. Does not mutate the graph."
    ),
    labels=frozenset({"env", "read", "setup", "operate"}),
)


def run_env_navix_maps(ctx: ToolContext, request: ToolRequest) -> str:
    """List Navix maps via ``jarl.operations.env.navix_maps``."""
    del ctx
    response = navix_maps(
        NavixMapsRequest(
            categories=request.categories,
            query=request.query,
            difficulty_min=request.difficulty_min,
            difficulty_max=request.difficulty_max,
            jarl_transfer_ready_only=request.jarl_transfer_ready_only,
            installed_only=request.installed_only,
        )
    )
    return json.dumps(response.to_compact_dict(), ensure_ascii=False, default=str)


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(workflow, run_env_navix_maps, TOOL_SPEC, request_cls=ToolRequest)
