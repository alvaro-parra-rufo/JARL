"""Agentic case for reading experiment state without mutation."""

from __future__ import annotations

from jarl.agentic.cases.base_agentic_case import (
    AgenticCaseRun,
    AgenticTurn,
    BaseAgenticCase,
)
from jarl.agentic.cases.validation import ValidationReport
from jarl.experiments.cases import CaseSpec
from jarl.experiments.cases.builtin.demo_tree import DemoTreeCase

__all__ = ["CASE", "GraphReadWithoutMutationCase"]

_EXPECTED_NODE_COUNT = 3
"""Node count materialized by `DemoTreeCase`."""


class GraphReadWithoutMutationCase(BaseAgenticCase):
    """Validate that an informational request only uses read tools."""

    def __init__(self) -> None:
        """Initialize the read-only case."""
        super().__init__(
            CaseSpec(
                id="graph_read_without_mutation",
                title="Read experiment state",
                description="Inspect a demo tree without changing its graph.",
                tags=frozenset({"graph", "operate", "read"}),
                requirements=frozenset({"llm"}),
            ),
            DemoTreeCase(),
            (
                AgenticTurn(
                    "Resume el estado del experimento y dime cuántos nodos tiene. "
                    "Es una consulta informativa: no modifiques el grafo ni entrenes."
                ),
            ),
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Validate read audit evidence and unchanged graph shape."""
        report.check(
            any(event.kind == "read" for event in run.audit_events),
            "Expected at least one audited read tool call.",
        )
        report.check(
            not any(event.kind in {"mutation", "train"} for event in run.audit_events),
            "Expected no mutation or training tool calls.",
            halted=True,
        )
        report.check(
            run.graph.as_networkx().number_of_nodes() == _EXPECTED_NODE_COUNT,
            "Expected the three-node demo tree to remain unchanged.",
        )
        report.check(
            set(run.graph.branch_heads) == {"main", "exp"},
            "Expected the original main and exp branches only.",
        )


CASE = GraphReadWithoutMutationCase()
"""Built-in read-only experiment case."""
