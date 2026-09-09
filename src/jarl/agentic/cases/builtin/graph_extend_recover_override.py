"""Agentic case for recovering from an invalid extend override."""

from __future__ import annotations

from jarl.agentic.audit import AuditIndexEvent
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

__all__ = ["CASE", "GraphExtendRecoverOverrideCase"]

_EXPECTED_EVALUATION_FREQUENCY = 128
"""Evaluation cadence requested by the recovery scenario."""

_EXPECTED_NODE_COUNT = 2
"""Root and one successfully recovered child."""


class GraphExtendRecoverOverrideCase(BaseAgenticCase):
    """Validate correction and retry after an invalid dotted override."""

    def __init__(self) -> None:
        """Initialize the override recovery case."""
        super().__init__(
            CaseSpec(
                id="graph_extend_recover_override",
                title="Recover an invalid extend override",
                description="Retry graph_extend with the canonical eval-frequency path.",
                tags=frozenset({"graph", "operate", "extend", "recovery"}),
                requirements=frozenset({"llm"}),
            ),
            NavixPreparedRootCase(),
            (
                AgenticTurn(
                    "Extiende la rama main con un nodo preparado llamado recovered y "
                    "cambia la frecuencia de evaluación a 128. Primero intenta "
                    "deliberadamente el override inválido `algorithm.eval_frequency`; "
                    "lee el error y reintenta con el path canónico. No entrenes."
                ),
            ),
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Validate failed alias use followed by a successful canonical retry."""
        invalid_indices = [
            index
            for index, event in enumerate(run.audit_events)
            if event.tool == "graph_extend"
            and event.kind == "error"
            and _override_value(event, "algorithm.eval_frequency") == _EXPECTED_EVALUATION_FREQUENCY
        ]
        recovered_indices = [
            index
            for index, event in enumerate(run.audit_events)
            if event.tool == "graph_extend"
            and event.kind == "mutation"
            and _override_value(
                event,
                "algorithm.evaluation_and_save_frequency",
            )
            == _EXPECTED_EVALUATION_FREQUENCY
        ]
        report.check(
            bool(invalid_indices),
            "Expected graph_extend to reject algorithm.eval_frequency.",
        )
        report.check(
            bool(recovered_indices),
            "Expected graph_extend to retry with the canonical evaluation path.",
        )
        mutations = [event.tool for event in run.audit_events if event.kind == "mutation"]
        report.check(
            all(tool == "graph_extend" for tool in mutations),
            "Expected no mutation other than graph_extend.",
            halted=True,
        )
        report.check(
            mutations.count("graph_extend") <= 1,
            "Expected at most one successful graph_extend mutation.",
            halted=True,
        )
        report.check(
            not invalid_indices or not recovered_indices or min(invalid_indices) < min(recovered_indices),
            "Expected the valid retry after the rejected override.",
        )
        report.check(
            run.graph.as_networkx().number_of_nodes() == _EXPECTED_NODE_COUNT,
            "Expected exactly one child after recovery.",
        )
        config = run.graph.resolve_config(run.graph.current_node)
        report.check(
            config.algorithm.evaluation_and_save_frequency == _EXPECTED_EVALUATION_FREQUENCY,
            "Expected the recovered node config to use evaluation frequency 128.",
        )
        report.check(
            not any(event.kind == "train" for event in run.audit_events),
            "Expected no training tool calls.",
            halted=True,
        )


def _override_value(event: AuditIndexEvent, path: str) -> object:
    """Return one dotted override value from an audited tool request."""
    overrides = event.request.get("config_overrides")
    if not isinstance(overrides, dict):
        return None
    return overrides.get(path)


CASE = GraphExtendRecoverOverrideCase()
"""Built-in invalid-override recovery case."""
