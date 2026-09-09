"""Agentic case for creating a baseline root during setup."""

from __future__ import annotations

from jarl.agentic.cases.base_agentic_case import (
    AgenticCaseRun,
    AgenticTurn,
    BaseAgenticCase,
)
from jarl.agentic.cases.validation import ValidationReport
from jarl.experiments.cases import CaseSpec
from jarl.experiments.cases.builtin.empty_experiment import EmptyExperimentCase

__all__ = ["CASE", "GraphCreateRootCase"]


class GraphCreateRootCase(BaseAgenticCase):
    """Validate root creation from an empty experiment."""

    def __init__(self) -> None:
        """Initialize the setup case."""
        super().__init__(
            CaseSpec(
                id="graph_create_root_baseline",
                title="Create a baseline root",
                description="Create one prepared baseline root without training.",
                tags=frozenset({"graph", "setup", "create_root"}),
                requirements=frozenset({"llm"}),
            ),
            EmptyExperimentCase(),
            (
                AgenticTurn(
                    "Inicializa un experimento baseline en la rama main con el preset fast. "
                    "Déjalo preparado y no entrenes."
                ),
            ),
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Validate the persisted root and tool audit."""
        node_count = run.graph.as_networkx().number_of_nodes()
        mutations = [event.tool for event in run.audit_events if event.kind == "mutation"]
        report.check(
            node_count == 1,
            "Expected exactly one root node.",
            halted=node_count > 1,
        )
        report.check(
            all(tool == "graph_create_root" for tool in mutations),
            "Expected no mutation other than graph_create_root.",
            halted=True,
        )
        report.check(
            mutations.count("graph_create_root") <= 1,
            "Expected at most one graph_create_root mutation.",
            halted=True,
        )
        report.check(
            any(event.tool == "graph_create_root" and event.kind == "mutation" for event in run.audit_events),
            "Expected a successful graph_create_root call.",
        )
        report.check(
            not any(event.kind == "train" for event in run.audit_events),
            "Expected no training tool calls.",
            halted=True,
        )


CASE = GraphCreateRootCase()
"""Built-in root creation case."""
