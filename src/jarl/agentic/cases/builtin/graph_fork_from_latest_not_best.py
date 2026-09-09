"""Agentic case: fork from ``latest`` when ``best`` also exists but is not wanted."""

from __future__ import annotations

from jarl.agentic.cases.base_agentic_case import (
    AgenticCaseRun,
    AgenticTurn,
    BaseAgenticCase,
)
from jarl.agentic.cases.validation import ValidationReport
from jarl.experiments.cases import CaseSpec
from jarl.experiments.cases.builtin.navix_root_best_vs_latest import (
    BEST_CHECKPOINT_STEP,
    LATEST_CHECKPOINT_STEP,
    NavixRootBestVsLatestCase,
)

__all__ = ["CASE", "GraphForkFromLatestNotBestCase"]

_EXPECTED_NODE_COUNT = 2
"""Root plus one forked child."""


class GraphForkFromLatestNotBestCase(BaseAgenticCase):
    """Validate forking from ``latest`` when ``best`` exists but is not the intent."""

    def __init__(self) -> None:
        """Initialize the latest-over-best fork case."""
        super().__init__(
            CaseSpec(
                id="graph_fork_from_latest_not_best",
                title="Fork from latest, not best",
                description=(
                    "Prefer latest over best when the user wants to continue from the most recent checkpoint."
                ),
                tags=frozenset({"graph", "operate", "fork", "checkpoints", "latest"}),
                requirements=frozenset({"llm", "navix"}),
            ),
            NavixRootBestVsLatestCase(),
            (
                AgenticTurn(
                    "Quiero probar una variante desde el checkpoint más reciente "
                    ", no desde el best: conviene continuar desde donde "
                    "quedó el entrenamiento aunque la eval sea peor. Crea una rama "
                    "preparada llamada from_latest con label latest_child. No entrenes."
                ),
            ),
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Validate fork uses latest, not the better-eval best alias."""
        fork_events = [event for event in run.audit_events if event.tool == "graph_fork" and event.kind == "mutation"]
        mutations = [event.tool for event in run.audit_events if event.kind == "mutation"]
        report.check(bool(fork_events), "Expected a successful graph_fork call.")
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
            any(event.tool == "graph_checkpoints" and event.kind == "read" for event in run.audit_events),
            "Expected graph_checkpoints before choosing a checkpoint.",
        )
        report.check(
            run.graph.as_networkx().number_of_nodes() == _EXPECTED_NODE_COUNT,
            "Expected root plus one forked child.",
        )
        if not fork_events:
            return
        mutation = fork_events[0]
        from_checkpoint = mutation.request.get("from_checkpoint")
        report.check(
            isinstance(from_checkpoint, dict),
            "Expected from_checkpoint on the graph_fork request.",
        )
        if not isinstance(from_checkpoint, dict):
            return
        report.check(
            from_checkpoint.get("checkpoint_step") == LATEST_CHECKPOINT_STEP,
            f"Expected from_checkpoint.checkpoint_step == {LATEST_CHECKPOINT_STEP} (latest).",
        )
        report.check(
            from_checkpoint.get("checkpoint_step") != BEST_CHECKPOINT_STEP,
            "Expected the agent not to select best when latest was requested.",
        )
        root_id = run.experiment.node_id("root")
        report.check(
            from_checkpoint.get("node_id") == root_id,
            "Expected from_checkpoint.node_id to be the prepared root.",
        )
        child = run.graph.current_node
        report.check(
            child.parent_checkpoint_step == LATEST_CHECKPOINT_STEP,
            f"Expected child parent_checkpoint_step == {LATEST_CHECKPOINT_STEP}.",
        )


CASE = GraphForkFromLatestNotBestCase()
"""Built-in fork-from-latest-not-best checkpoint case."""
