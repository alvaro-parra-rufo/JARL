"""LLM subagent that interprets objective metric features."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, field_validator, model_validator

from jarl.agentic.langgraph.graphs.deterministic import DeterministicSubagentState
from jarl.agentic.llm.structured_output import StructuredOutputMethod, resolve_structured_output_method

if TYPE_CHECKING:
    from jarl.agentic.tools.context import ToolContext
    from jarl.agentic.tools.specs import ToolRegistry
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = [
    "GRAPH_ID",
    "METRIC_ANALYSIS_LABELS",
    "METRIC_LABEL_SYNONYMS",
    "MetricsAnalysisOutput",
    "build_graph",
    "parse_metrics_analysis_payload",
    "validate_evidence_subset",
]

GRAPH_ID = "subagents/metrics_analysis"

METRIC_ANALYSIS_LABELS = (
    "improving",
    "stable",
    "degrading",
    "plateau",
    "high_variance",
    "late_regression",
    "possible_collapse",
    "insufficient_evidence",
)
"""Closed multietiqueta vocabulary for metrics-analysis outputs."""

MetricAnalysisLabel = Literal[
    "improving",
    "stable",
    "degrading",
    "plateau",
    "high_variance",
    "late_regression",
    "possible_collapse",
    "insufficient_evidence",
]

METRIC_LABEL_SYNONYMS: dict[str, MetricAnalysisLabel] = {
    "stalled": "plateau",
    "stagnant": "plateau",
    "flat": "plateau",
    "collapse": "possible_collapse",
    "collapsed": "possible_collapse",
    "collapsing": "possible_collapse",
    "regression": "late_regression",
    "regressing": "late_regression",
    "volatile": "high_variance",
    "noisy": "high_variance",
}
"""Map common LLM label mistakes to the closed vocabulary."""

_ALLOWED_LABELS_TEXT = ", ".join(METRIC_ANALYSIS_LABELS)

_SYSTEM_PROMPT = f"""\
You analyze RL training metric features for an experiment node.
Use only the feature ids and numeric values in the payload.
Return labels from the closed vocabulary, a short free-text summary, and evidence ids.
Allowed labels ONLY: {_ALLOWED_LABELS_TEXT}.
Never invent other label names (for example not stalled, converged, or collapsed).
Every evidence id must appear in the payload feature_ids list.
Do not invent metrics, steps, or values absent from the payload.
Prefer insufficient_evidence when features are too sparse to judge.
Respond with JSON only, using this shape:
{{"labels":["improving"],"summary":"...","evidence":["feature_id"]}}
"""


class MetricsAnalysisOutput(BaseModel):
    """Structured qualitative metrics analysis returned to the main agent."""

    labels: list[MetricAnalysisLabel] = Field(
        min_length=1,
        description="Closed multietiqueta labels describing the curves.",
    )
    summary: str = Field(
        min_length=1,
        max_length=600,
        description="Short free-text summary justified by evidence feature ids.",
    )
    evidence: list[str] = Field(
        min_length=1,
        description="Feature ids from the preprocess payload that justify the summary.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_labels(cls, data: object) -> object:
        """Normalize unknown or synonym labels before literal validation."""
        if not isinstance(data, dict):
            return data
        raw_labels = data.get("labels")
        if not isinstance(raw_labels, list):
            return data
        allowed = set(METRIC_ANALYSIS_LABELS)
        coerced: list[str] = []
        for item in raw_labels:
            label = str(item).strip()
            if not label:
                continue
            if label in allowed:
                coerced.append(label)
                continue
            mapped = METRIC_LABEL_SYNONYMS.get(label)
            if mapped is not None:
                coerced.append(mapped)
        if not coerced:
            coerced = ["insufficient_evidence"]
        return {**data, "labels": list(dict.fromkeys(coerced))}

    @field_validator("labels")
    @classmethod
    def _unique_labels(cls, value: list[MetricAnalysisLabel]) -> list[MetricAnalysisLabel]:
        return list(dict.fromkeys(value))

    @field_validator("evidence")
    @classmethod
    def _unique_evidence(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            msg = "evidence must contain at least one feature id."
            raise ValueError(msg)
        return list(dict.fromkeys(cleaned))


def parse_metrics_analysis_payload(payload: object) -> MetricsAnalysisOutput:
    """Parse and normalize LLM JSON into ``MetricsAnalysisOutput``."""
    return MetricsAnalysisOutput.model_validate(payload)


def validate_evidence_subset(output: MetricsAnalysisOutput, feature_ids: set[str]) -> None:
    """Raise ``ValueError`` when any evidence id is outside ``feature_ids``."""
    unknown = [item for item in output.evidence if item not in feature_ids]
    if unknown:
        msg = f"evidence contains unknown feature ids: {unknown}"
        raise ValueError(msg)


def build_graph(
    ctx: ToolContext,
    _registry: ToolRegistry,
    llm: BaseChatModel | None,
) -> CompiledStateGraph:
    """Build a single-node LangGraph that interprets metric features with an LLM."""
    if llm is None:
        msg = "subagents/metrics_analysis requires an LLM."
        raise ValueError(msg)

    method = _structured_output_method(ctx)

    def analyze_node(state: DeterministicSubagentState) -> DeterministicSubagentState:
        payload = state["input"]
        feature_ids = payload.get("feature_ids", [])
        if not isinstance(feature_ids, list):
            feature_ids = []
        human = (
            f"node_id={payload.get('node_id')}\n"
            f"focus={payload.get('focus', '')}\n"
            f"feature_ids={feature_ids}\n"
            f"features={payload.get('features', {})}\n"
        )
        messages: list[BaseMessage] = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human),
        ]
        output = _invoke_metrics_analysis_llm(llm, method, messages)
        validate_evidence_subset(output, {str(item) for item in feature_ids})
        return {
            "input": state["input"],
            "output": output.model_dump(exclude_none=True),
        }

    graph = StateGraph(DeterministicSubagentState)
    graph.add_node("analyze", analyze_node)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", END)
    return graph.compile()


def _invoke_metrics_analysis_llm(
    llm: BaseChatModel,
    method: StructuredOutputMethod,
    messages: list[BaseMessage],
) -> MetricsAnalysisOutput:
    """Invoke the LLM and parse metrics-analysis JSON with label coercion."""
    if method == "json_schema":
        result = llm.with_structured_output(MetricsAnalysisOutput, method=method).invoke(messages)
        if isinstance(result, MetricsAnalysisOutput):
            return result
        return parse_metrics_analysis_payload(result)

    response = llm.bind(response_format={"type": "json_object"}).invoke(messages)
    content = response.content
    if not isinstance(content, str):
        msg = "Metrics-analysis LLM response must be a JSON string."
        raise TypeError(msg)
    return parse_metrics_analysis_payload(json.loads(content))


def _structured_output_method(ctx: ToolContext) -> StructuredOutputMethod:
    """Resolve structured-output mode from the workflow's last LLM resolution."""
    workflow: AgenticWorkflow = ctx.workflow
    resolved = workflow.last_resolved_llm
    if resolved is None:
        return "json_mode"
    return resolve_structured_output_method(resolved.settings)
