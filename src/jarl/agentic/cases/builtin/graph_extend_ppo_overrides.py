"""Agentic case for extend with PPO hyperparameter overrides."""

from __future__ import annotations

from jarl.agentic.cases.base_agentic_case import (
    AgenticCaseRun,
    AgenticTurn,
    BaseAgenticCase,
)
from jarl.agentic.cases.builtin._expected_ppo_overrides import EXPECTED_PPO_OVERRIDES
from jarl.agentic.cases.validation import ValidationReport
from jarl.experiments.cases import CaseSpec
from jarl.experiments.cases.builtin.navix_prepared_root import (
    NavixPreparedRootCase,
)

__all__ = ["CASE", "GraphExtendPpoOverridesCase"]

_EXPECTED_NODE_COUNT = 2
"""Root and one child with overridden PPO hyperparameters."""


class GraphExtendPpoOverridesCase(BaseAgenticCase):
    """Validate extend with several PPO ``config_overrides`` applied."""

    def __init__(self) -> None:
        """Initialize the PPO override extension case."""
        super().__init__(
            CaseSpec(
                id="graph_extend_ppo_overrides",
                title="Extend with PPO overrides",
                description=("Extend the current branch with PPO hyperparameter overrides on a prepared child."),
                tags=frozenset({"graph", "operate", "extend", "config_overrides", "ppo"}),
                requirements=frozenset({"llm"}),
            ),
            NavixPreparedRootCase(),
            (
                AgenticTurn(
                    "Extiende la rama main con un hijo preparado label tuned y estos "
                    "hiperparámetros PPO: learning rate 0.001, gamma 0.97, entropy "
                    "coefficient 0.08, GAE lambda 0.9 y clip range 0.15. No entrenes."
                ),
            ),
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Validate extend mutation and merged PPO hyperparameters."""
        root_id = run.experiment.node_id("root")
        current = run.graph.current_node

        extend_events = [
            event for event in run.audit_events if event.tool == "graph_extend" and event.kind == "mutation"
        ]
        report.check(
            extend_events,
            "Expected graph_extend to create the tuned child.",
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
            "Expected root plus one tuned child.",
        )
        report.check(current.branch == "main", "Expected the child to remain on main.")
        report.check(
            current.node_metadata.parent_id == root_id,
            "Expected the original root to be the child parent.",
        )
        if not extend_events:
            return

        mutation = extend_events[0]
        overrides = mutation.request.get("config_overrides")
        report.check(
            isinstance(overrides, dict),
            "Expected config_overrides on the graph_extend request.",
        )
        if isinstance(overrides, dict):
            for path, expected in EXPECTED_PPO_OVERRIDES.items():
                report.check(
                    overrides.get(path) == expected,
                    f"Expected audited override {path}={expected!r}.",
                )

        config = run.graph.resolve_config(current)
        report.check(
            config.algorithm.learning_rate == EXPECTED_PPO_OVERRIDES["algorithm.learning_rate"],
            "Expected tuned learning rate on the child config.",
        )
        report.check(
            config.algorithm.gamma == EXPECTED_PPO_OVERRIDES["algorithm.gamma"],
            "Expected tuned gamma on the child config.",
        )
        report.check(
            config.algorithm.entropy_coef == EXPECTED_PPO_OVERRIDES["algorithm.entropy_coef"],
            "Expected tuned entropy coefficient on the child config.",
        )
        report.check(
            config.algorithm.gae_lambda == EXPECTED_PPO_OVERRIDES["algorithm.gae_lambda"],
            "Expected tuned GAE lambda on the child config.",
        )
        report.check(
            config.algorithm.clip_range == EXPECTED_PPO_OVERRIDES["algorithm.clip_range"],
            "Expected tuned clip range on the child config.",
        )


CASE = GraphExtendPpoOverridesCase()
"""Built-in extend case with PPO hyperparameter overrides."""
