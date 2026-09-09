"""Agentic case for choosing extend on the current branch."""

from __future__ import annotations

from jarl.agentic.cases.base_agentic_case import (
    AgenticCaseRun,
    AgenticTurn,
    BaseAgenticCase,
)
from jarl.agentic.cases.validation import ValidationReport
from jarl.experiments.cases import CaseSpec
from jarl.experiments.cases.builtin.navix_prepared_root import (
    NavixPreparedRootCase,
)

__all__ = ["CASE", "GraphExtendSameBranchCase"]

_EXPECTED_NODE_COUNT = 2
"""Root and one extended child."""


class GraphExtendSameBranchCase(BaseAgenticCase):
    """Validate that same-line continuation uses `graph_extend`."""

    def __init__(self) -> None:
        """Initialize the same-branch extension case."""
        super().__init__(
            CaseSpec(
                id="graph_extend_same_branch",
                title="Extend the current branch",
                description="Continue main with one prepared child and no parallel branch.",
                tags=frozenset({"graph", "operate", "extend", "desambiguation"}),
                requirements=frozenset({"llm"}),
            ),
            NavixPreparedRootCase(),
            (
                AgenticTurn(
                    "Continúa la línea actual de la rama main creando un hijo preparado "
                    "con label continued. No abras una rama paralela y no entrenes."
                ),
            ),
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Validate extend selection and resulting lineage."""
        root_id = run.experiment.node_id("root")
        current = run.graph.current_node
        report.check(
            any(event.tool == "graph_extend" and event.kind == "mutation" for event in run.audit_events),
            "Expected a successful graph_extend call.",
        )
        mutations = [event.tool for event in run.audit_events if event.kind == "mutation"]
        report.check(
            all(tool == "graph_extend" for tool in mutations),
            "Expected no mutation other than graph_extend.",
            halted=True,
        )
        report.check(
            mutations.count("graph_extend") <= 1,
            "Expected at most one graph_extend mutation.",
            halted=True,
        )
        report.check(
            not any(event.kind == "train" for event in run.audit_events),
            "Expected no training tool calls.",
            halted=True,
        )
        report.check(
            run.graph.as_networkx().number_of_nodes() == _EXPECTED_NODE_COUNT,
            "Expected root plus one child node.",
        )
        report.check(current.branch == "main", "Expected the child to remain on main.")
        report.check(
            current.node_metadata.parent_id == root_id,
            "Expected the original root to be the child parent.",
        )


CASE = GraphExtendSameBranchCase()
"""Built-in same-branch extension case."""
