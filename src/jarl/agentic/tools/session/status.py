"""Report the active agent session and experiment branch state."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.handler import wire_handler
from jarl.agentic.tools.specs import ToolSpec
from jarl.experiments.graph import ExperimentGraph
from jarl.training.config import RLRunConfig

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_session_status"]


class ToolRequest(BaseModel):
    """Empty request for workflow status."""


TOOL_SPEC = ToolSpec(
    name="session_status",
    description=("Read thread id, active node, branch heads, and current-branch nodes. Does not mutate the graph."),
    labels=frozenset({"session", "read", "setup", "operate"}),
)


def run_session_status(ctx: ToolContext, request: ToolRequest) -> str:
    """Return a compact JSON snapshot of the active session and branch."""
    del request
    graph = ctx.graph
    active_node_id = ctx.try_current_node_id()
    active_status: str | None = None
    current_branch: str | None = None
    current_branch_summary: dict[str, object] | None = None
    if active_node_id is not None:
        workspace = graph.get_node(active_node_id)
        active_status = workspace.status.value
        current_branch = workspace.branch
        current_branch_summary = _branch_summary(graph, current_branch)
    payload = {
        "thread_id": ctx.thread_id,
        "active_node_id": active_node_id,
        "active_status": active_status,
        "node_count": len(graph.all_nodes),
        "current_branch": current_branch,
        "current_branch_summary": current_branch_summary,
        "branch_heads": {branch: workspace.id for branch, workspace in graph.branch_heads.items()},
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _branch_summary(graph: ExperimentGraph[RLRunConfig], branch: str) -> dict[str, object]:
    nodes = [
        {
            "node_id": workspace.id,
            "status": workspace.status.value,
            "label": workspace.node_metadata.label,
        }
        for workspace in graph.all_nodes.values()
        if workspace.branch == branch
    ]
    return {
        "branch": branch,
        "head_id": graph.branch_heads[branch].id,
        "nodes": nodes,
    }


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(workflow, run_session_status, TOOL_SPEC, request_cls=ToolRequest)
