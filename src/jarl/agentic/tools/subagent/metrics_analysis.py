"""Analyze metric curves via preprocess + LLM subagent."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from jarl.agentic.langgraph.graphs.subagents.metrics_analysis import (
    GRAPH_ID,
    MetricsAnalysisOutput,
    validate_evidence_subset,
)
from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.handler import wire_handler
from jarl.agentic.tools.specs import ToolSpec
from jarl.operations.subagent.metrics_analysis import MetricsAnalysisRequest, metrics_analysis
from jarl.utils.pydantic import FlexibleList

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["TOOL_SPEC", "ToolRequest", "build_handler", "run_subagent_metrics_analysis"]


class ToolRequest(BaseModel):
    """Inputs for qualitative metrics analysis on a node."""

    node_id: str | None = Field(
        default=None,
        description="Node whose metric curves to analyze. Omit for the current node.",
    )
    metric_keys: FlexibleList[str] | None = Field(
        default=None,
        description=(
            "Exact metric names from the node metrics.jsonl. Omit for the default "
            "rollout/eval set. Unknown keys raise an error listing available names."
        ),
    )
    focus: str = Field(
        default="",
        description="Optional question that steers the qualitative interpretation.",
    )


TOOL_SPEC = ToolSpec(
    name="subagent_metrics_analysis",
    description=(
        "Analyze metric curves for a node via a subagent. Returns closed labels, "
        "a short summary, and evidence feature ids. Not for checkpoint ranking or "
        "recovery diagnosis. Read-only; does not mutate the graph."
    ),
    labels=frozenset({"subagent", "graph", "read", "operate"}),
)


def run_subagent_metrics_analysis(ctx: ToolContext, request: ToolRequest) -> str:
    """Build objective features and invoke the metrics-analysis subagent."""
    node_id = request.node_id or ctx.try_current_node_id()
    if node_id is None:
        msg = "node_id is required when no active node is set."
        raise ValueError(msg)
    features = metrics_analysis(
        ctx.graph,
        MetricsAnalysisRequest(
            node_id=node_id,
            metric_keys=request.metric_keys,
            focus=request.focus,
        ),
    )
    raw = ctx.invoke_subgraph(
        GRAPH_ID,
        features.to_compact_dict(),
        parent_tool=TOOL_SPEC.name,
        thread_suffix="metrics_analysis",
    )
    output = MetricsAnalysisOutput.model_validate(raw)
    validate_evidence_subset(output, set(features.feature_ids))
    return json.dumps(output.model_dump(exclude_none=True), ensure_ascii=False, default=str)


def build_handler(workflow: AgenticWorkflow) -> Callable[..., str]:
    """Build the LangGraph handler for this tool."""
    return wire_handler(
        workflow,
        run_subagent_metrics_analysis,
        TOOL_SPEC,
        request_cls=ToolRequest,
    )
