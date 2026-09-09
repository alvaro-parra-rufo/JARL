"""Agentic case for choosing fork for a parallel branch."""

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

__all__ = ["CASE", "GraphForkParallelBranchCase"]

_EXPECTED_NODE_COUNT = 2
"""Root and one forked child."""


class GraphForkParallelBranchCase(BaseAgenticCase):
    """Validate that a parallel variant uses `graph_fork`."""

    def __init__(self) -> None:
        """Initialize the parallel-branch fork case."""
        super().__init__(
            CaseSpec(
                id="graph_fork_parallel_branch",
                title="Fork a parallel branch",
                description="Create one prepared child on a new exploration branch.",
                tags=frozenset({"graph", "operate", "fork", "desambiguation"}),
                requirements=frozenset({"llm"}),
            ),
            NavixPreparedRootCase(),
            (
                AgenticTurn(
                    "Crea una variante paralela preparada en una rama nueva llamada "
                    "exploration desde el nodo actual. No continúes main y no entrenes."
                ),
            ),
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Validate fork selection and the new branch head."""
        mutations = [event.tool for event in run.audit_events if event.kind == "mutation"]
        report.check(
            any(event.tool == "graph_fork" and event.kind == "mutation" for event in run.audit_events),
            "Expected a successful graph_fork call.",
        )
        report.check(
            all(tool == "graph_fork" for tool in mutations),
            "Expected no mutation other than graph_fork.",
            halted=True,
        )
        report.check(
            mutations.count("graph_fork") <= 1,
            "Expected at most one graph_fork mutation.",
            halted=True,
        )
        report.check(
            not any(event.kind == "train" for event in run.audit_events),
            "Expected no training tool calls.",
            halted=True,
        )
        report.check(
            run.graph.as_networkx().number_of_nodes() == _EXPECTED_NODE_COUNT,
            "Expected root plus one forked child.",
        )
        report.check(
            "exploration" in run.graph.branch_heads,
            "Expected an exploration branch.",
        )
        report.check(
            run.graph.current_node.branch == "exploration",
            "Expected the forked exploration node to be current.",
        )


CASE = GraphForkParallelBranchCase()
"""Built-in parallel-branch fork case."""
