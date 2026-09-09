"""Agentic case for fork with PPO hyperparameter overrides."""

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

__all__ = ["CASE", "GraphForkPpoOverridesCase"]

_EXPECTED_BRANCH = "ppo_tuned"
"""New branch created by the fork scenario (distinct from extend's main-line label tuned)."""

_EXPECTED_NODE_COUNT = 2
"""Root on main and one tuned child on the forked branch."""


class GraphForkPpoOverridesCase(BaseAgenticCase):
    """Validate fork with the same PPO ``config_overrides`` as the extend scenario."""

    def __init__(self) -> None:
        """Initialize the PPO override fork case."""
        super().__init__(
            CaseSpec(
                id="graph_fork_ppo_overrides",
                title="Fork with PPO overrides",
                description=("Fork a new branch from root with PPO hyperparameter overrides on a prepared child."),
                tags=frozenset({"graph", "operate", "fork", "config_overrides", "ppo"}),
                requirements=frozenset({"llm"}),
            ),
            NavixPreparedRootCase(),
            (
                AgenticTurn(
                    "Crea una rama nueva llamada ppo_tuned desde el nodo root con un hijo "
                    "preparado label tuned_fork y estos hiperparámetros PPO: learning rate "
                    "0.001, gamma 0.97, entropy coefficient 0.08, GAE lambda 0.9 y clip "
                    "range 0.15. No uses extend ni entrenes."
                ),
            ),
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Validate fork mutation and merged PPO hyperparameters."""
        root_id = run.experiment.node_id("root")
        current = run.graph.current_node

        fork_events = [event for event in run.audit_events if event.tool == "graph_fork" and event.kind == "mutation"]
        mutations = [event.tool for event in run.audit_events if event.kind == "mutation"]
        report.check(
            fork_events,
            "Expected graph_fork to create the tuned child.",
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
            "Expected root plus one tuned fork child.",
        )
        report.check(
            _EXPECTED_BRANCH in run.graph.branch_heads,
            f"Expected a {_EXPECTED_BRANCH!r} branch.",
        )
        report.check(
            current.branch == _EXPECTED_BRANCH,
            f"Expected the forked child to be current on {_EXPECTED_BRANCH!r}.",
        )
        report.check(
            current.node_metadata.parent_id == root_id,
            "Expected the root node to be the fork parent.",
        )
        if not fork_events:
            return

        mutation = fork_events[0]
        overrides = mutation.request.get("config_overrides")
        report.check(
            isinstance(overrides, dict),
            "Expected config_overrides on the graph_fork request.",
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


CASE = GraphForkPpoOverridesCase()
"""Built-in fork case with PPO hyperparameter overrides."""
