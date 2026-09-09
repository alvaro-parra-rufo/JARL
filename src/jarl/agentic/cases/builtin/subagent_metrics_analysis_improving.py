"""Agentic case: qualitative metrics analysis on an improving eval curve."""

from __future__ import annotations

from jarl.agentic.cases.base_agentic_case import (
    AgenticCaseRun,
    AgenticTurn,
    BaseAgenticCase,
)
from jarl.agentic.cases.validation import ValidationReport
from jarl.experiments.cases import CaseSpec
from jarl.experiments.cases.builtin.navix_root_improving_eval import NavixRootImprovingEvalCase

__all__ = ["CASE", "SubagentMetricsAnalysisImprovingCase"]


class SubagentMetricsAnalysisImprovingCase(BaseAgenticCase):
    """Validate metrics-analysis routing and structured evidence output."""

    def __init__(self) -> None:
        """Initialize the metrics-analysis improving-curve case."""
        super().__init__(
            CaseSpec(
                id="subagent_metrics_analysis_improving",
                title="Analyze improving eval metrics",
                description="Call subagent_metrics_analysis on a node with rising eval return.",
                tags=frozenset({"subagent", "operate", "read", "metrics"}),
                requirements=frozenset({"llm", "navix"}),
            ),
            NavixRootImprovingEvalCase(),
            (
                AgenticTurn(
                    "Analiza las curvas de evaluación de este nodo y dime si el "
                    "rendimiento mejora. No mutes el grafo ni entrenes."
                ),
            ),
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Validate subagent metrics analysis audit payload."""
        events = [
            event
            for event in run.audit_events
            if event.tool == "subagent_metrics_analysis" and event.kind == "subagent"
        ]
        report.check(bool(events), "Expected a successful subagent_metrics_analysis call.")
        report.check(
            not any(event.kind in {"mutation", "train"} for event in run.audit_events),
            "Expected no mutation or training tool calls.",
            halted=True,
        )
        if not events:
            return
        event = events[0]
        result = event.result or {}
        report.check(isinstance(result.get("labels"), list) and bool(result["labels"]), "Expected labels.")
        report.check(isinstance(result.get("summary"), str) and bool(result["summary"]), "Expected summary.")
        evidence = result.get("evidence")
        report.check(isinstance(evidence, list) and bool(evidence), "Expected evidence feature ids.")
        report.check(
            event.sub_thread_id is not None and event.sub_thread_id != event.thread_id,
            "Expected nested sub_thread_id distinct from parent thread_id.",
        )
        report.check(
            isinstance(event.run_dir, str) and "/subagents/metrics_analysis" in (event.run_dir or ""),
            "Expected nested run_dir under parent/subagents/metrics_analysis.",
        )
        if isinstance(evidence, list):
            report.check(
                all(isinstance(item, str) and item.startswith("eval_return.") for item in evidence),
                "Expected evidence ids from eval_return features.",
            )


CASE = SubagentMetricsAnalysisImprovingCase()
"""Built-in improving-metrics analysis case."""
