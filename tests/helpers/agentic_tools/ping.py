"""Sample read tool for registry discovery tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.handler import wire_handler
from jarl.agentic.tools.specs import ToolSpec

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_sample_ping"]


class ToolRequest(BaseModel):
    """Ping payload."""

    text: str = Field(default="ping")


TOOL_SPEC = ToolSpec(
    name="sample_ping",
    description="Sample read tool used in registry unit tests.",
    labels=frozenset({"graph", "read"}),
)


def run_sample_ping(ctx: ToolContext, request: ToolRequest) -> str:
    """Echo the request text and current node id."""
    return json.dumps({"text": request.text, "node_id": ctx.current_node_id})


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(workflow, run_sample_ping, TOOL_SPEC, request_cls=ToolRequest)
