"""Tests for the failed-resumable Navix root experiment case."""

from __future__ import annotations

from pathlib import Path

from jarl.experiments.cases.builtin.navix_root_failed_resumable import (
    CASE,
    RESUMABLE_CHECKPOINT_STEP,
)
from jarl.experiments.node import NodeStatus
from jarl.operations.train.recovery_status import RecoveryStatusRequest, recovery_status


def test_materialize_failed_resumable_root(tmp_path: Path) -> None:
    context = CASE.materialize(tmp_path / "case")
    graph = context.reload_graph()
    workspace = graph.get_node(context.node_id("root"))

    assert workspace.status == NodeStatus.FAILED
    resume = workspace.resolve_resume_checkpoint_record()
    assert resume is not None
    assert resume.checkpoint_step == RESUMABLE_CHECKPOINT_STEP

    status = recovery_status(graph, RecoveryStatusRequest(node_id=workspace.id))
    assert status.recommended_action == "train_resume"
    assert status.resumable_checkpoint_step == RESUMABLE_CHECKPOINT_STEP
