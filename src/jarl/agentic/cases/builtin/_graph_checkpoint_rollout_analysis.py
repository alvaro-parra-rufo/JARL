"""Shared helpers for the checkpoint rollout analysis agentic case."""

from __future__ import annotations

from jarl.agentic.cases.base_agentic_case import AgenticCaseRun
from jarl.agentic.cases.launch_options import CaseLaunchOption
from jarl.agentic.cases.validation import ValidationReport
from jarl.experiments.node import NodeStatus
from jarl.operations.contracts.constants import CHECKPOINT_ROLLOUT_RECORD_VIDEO_DEFAULT

__all__ = [
    "RECORD_VIDEO_LAUNCH_OPTION",
    "validate_checkpoint_rollout_analysis",
]

RECORD_VIDEO_LAUNCH_OPTION = CaseLaunchOption(
    key="record_video",
    label="Grabar vídeo MP4",
    description="Fuerza graph_checkpoint_rollout con record_video durante el caso.",
    tool="graph_checkpoint_rollout",
    field="record_video",
    option_type="bool",
    default=CHECKPOINT_ROLLOUT_RECORD_VIDEO_DEFAULT,
)


def validate_checkpoint_rollout_analysis(
    run: AgenticCaseRun,
    report: ValidationReport,
    *,
    checkpoint_step: int,
    seed: int,
    expected_node_status: NodeStatus = NodeStatus.COMPLETED,
    min_return: float | None = None,
    max_return: float | None = None,
) -> None:
    """Validate rollout read audit, summary payload, and optional return band."""
    rollout_events = [
        event for event in run.audit_events if event.tool == "graph_checkpoint_rollout" and event.kind == "read"
    ]
    report.check(bool(rollout_events), "Expected a successful graph_checkpoint_rollout read.")
    report.check(
        not any(event.kind in {"mutation", "train"} for event in run.audit_events),
        "Expected no mutation or training tool calls.",
        halted=True,
    )
    report.check(
        run.graph.as_networkx().number_of_nodes() == 1,
        "Expected the single trained root to remain unchanged.",
    )
    root = run.graph.get_node(run.experiment.node_id("root"))
    report.check(
        root.status is expected_node_status,
        f"Expected root node status {expected_node_status.value}, got {root.status.value}.",
    )
    if not rollout_events:
        return

    event = rollout_events[0]
    request = event.request
    report.check(
        request.get("checkpoint_step") == checkpoint_step,
        "Expected checkpoint_step in request.",
    )
    report.check(request.get("seed") == seed, "Expected rollout seed in request.")

    result = event.result or {}
    report.check(result.get("cache_hit") is False, "Expected first rollout execution, not cache hit.")
    identity = result.get("identity")
    if isinstance(identity, dict):
        report.check(identity.get("seed") == seed, "Expected summary identity seed.")
        report.check(
            identity.get("checkpoint_step") == checkpoint_step,
            "Expected summary identity checkpoint step.",
        )
    outcome = result.get("outcome")
    if isinstance(outcome, dict):
        report.check("length" in outcome, "Expected rollout length in outcome.")
        report.check("return_total" in outcome, "Expected rollout return in outcome.")
        report.check("reason" in outcome, "Expected rollout termination reason.")
        return_total = outcome.get("return_total")
        if isinstance(return_total, (int, float)):
            value = float(return_total)
            if min_return is not None:
                report.check(
                    value >= min_return,
                    f"Expected rollout return >= {min_return}, got {value}.",
                )
            if max_return is not None:
                report.check(
                    value <= max_return,
                    f"Expected rollout return <= {max_return}, got {value}.",
                )

    if expected_node_status is NodeStatus.COMPLETED:
        checkpoint_eval = result.get("checkpoint_eval")
        report.check(isinstance(checkpoint_eval, dict), "Expected checkpoint_eval payload.")
        if isinstance(checkpoint_eval, dict):
            report.check(
                "episode_return" in checkpoint_eval,
                "Expected checkpoint_eval.episode_return.",
            )
            report.check(
                "episode_length" in checkpoint_eval,
                "Expected checkpoint_eval.episode_length.",
            )

    artifact_paths = result.get("artifact_paths")
    if isinstance(artifact_paths, list):
        root = run.graph.get_node(run.experiment.node_id("root"))
        for relative_path in artifact_paths:
            if not isinstance(relative_path, str):
                report.check(False, f"Expected string artifact path, got {relative_path!r}.")
                continue
            artifact = root.path / relative_path
            report.check(artifact.is_file(), f"Expected persisted artifact at {relative_path}.")
