"""Declare that the current objective is already complete."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.handler import wire_handler
from jarl.agentic.tools.specs import ToolSpec
from jarl.operations.session.finish import FinishRequest, finish

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_session_finish"]


class ToolRequest(BaseModel):
    """Empty request for declaring finish."""


TOOL_SPEC = ToolSpec(
    name="session_finish",
    description=(
        "Call only when the current objective is already complete. Declares finish. "
        "Does not mutate the graph. Not for reading session state."
    ),
    labels=frozenset({"session", "finish", "setup", "operate"}),
)


def run_session_finish(ctx: ToolContext, request: ToolRequest) -> str:
    """Return a compact JSON declaration that the session is finished."""
    del request
    response = finish(FinishRequest(), thread_id=ctx.thread_id)
    return json.dumps(response.to_compact_dict(), ensure_ascii=False, default=str)


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(workflow, run_session_finish, TOOL_SPEC, request_cls=ToolRequest)
