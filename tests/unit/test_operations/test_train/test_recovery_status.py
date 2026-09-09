"""Tests for train recovery_status operation."""

from __future__ import annotations

import pytest

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeStatus
from jarl.operations.train.recovery_status import RecoveryStatusRequest, recovery_status
from jarl.training.config import RLRunConfig


def _fail_with_checkpoint(graph: ExperimentGraph[RLRunConfig], step: int = 5) -> None:
    workspace = graph.current_node
    workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
    try:
        with workspace:
            workspace.save_checkpoint(
                step,
                {"value": step},
                metrics={"eval/episode_return": 1.0},
                force=True,
            )
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    graph.save()


def _fail_without_checkpoint(graph: ExperimentGraph[RLRunConfig]) -> None:
    workspace = graph.current_node
    try:
        with workspace:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    graph.save()


class TestRecoveryStatus:
    def test_failed_with_checkpoint_recommends_train_resume(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        _fail_with_checkpoint(prepared_root, step=7)

        response = recovery_status(prepared_root, RecoveryStatusRequest())

        compact = response.to_compact_dict()
        assert response.node_status == NodeStatus.FAILED.value
        assert response.resumable_checkpoint_step == 7
        assert response.can_resume is True
        assert response.recommended_action == "train_resume"
        assert response.latest_attempt is not None
        assert response.latest_attempt.status == "failed"
        assert response.latest_attempt.error_type == "RuntimeError"
        assert "resume_hint" not in compact
        assert "latest" not in compact
        assert "best" not in compact
        assert "labels" not in compact

    def test_failed_without_checkpoint_recommends_none(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        _fail_without_checkpoint(prepared_root)

        response = recovery_status(prepared_root, RecoveryStatusRequest())

        assert response.node_status == NodeStatus.FAILED.value
        assert response.resumable_checkpoint_step is None
        assert response.can_resume is False
        assert response.recommended_action == "none"

    def test_prepared_node_recommends_none(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        response = recovery_status(prepared_root, RecoveryStatusRequest())

        assert response.node_status == NodeStatus.PREPARED.value
        assert response.can_resume is False
        assert response.recommended_action == "none"
        assert response.latest_attempt is None

    def test_explicit_node_id(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        node_id = prepared_root.current_node.id
        _fail_with_checkpoint(prepared_root)

        response = recovery_status(
            prepared_root,
            RecoveryStatusRequest(node_id=node_id),
        )

        assert response.node_id == node_id
        assert response.recommended_action == "train_resume"

    def test_missing_node_raises(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        with pytest.raises(KeyError):
            recovery_status(prepared_root, RecoveryStatusRequest(node_id="missing"))
