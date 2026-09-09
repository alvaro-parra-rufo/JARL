"""Agentic case: diagnose failed training and recommend train_resume."""

from __future__ import annotations

from jarl.agentic.cases.base_agentic_case import (
    AgenticCaseRun,
    AgenticTurn,
    BaseAgenticCase,
)
from jarl.agentic.cases.validation import ValidationReport
from jarl.experiments.cases import CaseSpec
from jarl.experiments.cases.builtin.navix_root_failed_resumable import (
    RESUMABLE_CHECKPOINT_STEP,
    NavixRootFailedResumableCase,
)
from jarl.experiments.node import NodeStatus

__all__ = ["CASE", "TrainRecoveryStatusResumeCase"]


class TrainRecoveryStatusResumeCase(BaseAgenticCase):
    """Validate recovery diagnosis for a failed node with a resumable checkpoint."""

    def __init__(self) -> None:
        """Initialize the train recovery diagnosis case."""
        super().__init__(
            CaseSpec(
                id="train_recovery_status_resume",
                title="Diagnose failed run for resume",
                description=("Read train_recovery_status on a failed node with a restorable checkpoint."),
                tags=frozenset({"train", "operate", "read", "recovery", "resume"}),
                requirements=frozenset({"llm", "navix"}),
            ),
            NavixRootFailedResumableCase(),
            (
                AgenticTurn(
                    "El entrenamiento de este nodo falló. Diagnostica si se puede "
                    "reanudar y qué acción estructurada corresponde. No reanudes "
                    "aún ni mutes el grafo."
                ),
            ),
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Validate recovery read and structured train_resume recommendation."""
        recovery_events = [
            event for event in run.audit_events if event.tool == "train_recovery_status" and event.kind == "read"
        ]
        report.check(bool(recovery_events), "Expected a successful train_recovery_status read.")
        report.check(
            not any(event.kind in {"mutation", "train"} for event in run.audit_events),
            "Expected no mutation or training tool calls.",
            halted=True,
        )
        root = run.graph.get_node(run.experiment.node_id("root"))
        report.check(
            root.status == NodeStatus.FAILED,
            "Expected the failed root to remain failed.",
        )
        if not recovery_events:
            return
        result = recovery_events[0].result or {}
        report.check(
            result.get("recommended_action") == "train_resume",
            "Expected recommended_action=train_resume.",
        )
        report.check(
            result.get("can_resume") is True,
            "Expected can_resume=true.",
        )
        report.check(
            result.get("resumable_checkpoint_step") == RESUMABLE_CHECKPOINT_STEP,
            f"Expected resumable_checkpoint_step == {RESUMABLE_CHECKPOINT_STEP}.",
        )
        report.check(
            "resume_hint" not in result,
            "Expected no resume_hint narrative field.",
        )


CASE = TrainRecoveryStatusResumeCase()
"""Built-in train recovery diagnosis case."""
