"""Sample mutation tool for registry filter tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.handler import wire_handler
from jarl.agentic.tools.specs import ToolSpec

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_sample_mutate"]


class ToolRequest(BaseModel):
    """Empty mutation request for filter tests."""


TOOL_SPEC = ToolSpec(
    name="sample_mutate",
    description="Sample mutation tool used in registry unit tests.",
    labels=frozenset({"graph", "mutation"}),
)


def run_sample_mutate(ctx: ToolContext, request: ToolRequest) -> str:
    """Return a static mutation acknowledgement."""
    del ctx, request
    return json.dumps({"status": "ok"})


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(workflow, run_sample_mutate, TOOL_SPEC, request_cls=ToolRequest)
