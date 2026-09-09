"""Tests for checkpoint pin and promote operations."""

from __future__ import annotations

import json

import pytest

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import (
    CHECKPOINT_ALIAS_BEST,
    CHECKPOINT_BEST_LENGTH_METRIC,
    CHECKPOINT_BEST_RETURN_METRIC,
    CheckpointRef,
)
from jarl.operations.graph.extend import ExtendRequest, extend
from jarl.operations.graph.fork import ForkRequest, fork
from jarl.operations.graph.pin import PinRequest, pin
from jarl.operations.graph.promote import PromoteRequest, promote
from jarl.training.config import RLRunConfig


class TestPinPromote:
    def test_pin_checkpoint(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        workspace = prepared_root.current_node
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        workspace.save_checkpoint(2, {"value": 2}, force=True)
        prepared_root.save()

        response = pin(
            prepared_root,
            PinRequest(node_id=workspace.id, checkpoint_step=2),
        )

        payload = json.loads(workspace.checkpoints_registry_path.read_text(encoding="utf-8"))

        assert response.pinned is True
        assert 2 in payload["pinned_checkpoint_steps"]

    def test_promote_best_checkpoint(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        workspace = prepared_root.current_node
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        workspace.save_checkpoint(1, {"value": 1}, metrics={CHECKPOINT_BEST_RETURN_METRIC: 0.5}, force=True)
        workspace.save_checkpoint(2, {"value": 2}, metrics={CHECKPOINT_BEST_RETURN_METRIC: 1.5}, force=True)
        prepared_root.save()

        response = promote(
            prepared_root,
            PromoteRequest(node_id=workspace.id),
        )

        assert response.checkpoint_step == 2
        assert response.metric_value == 1.5
        assert workspace.resolve_checkpoint_alias(CHECKPOINT_ALIAS_BEST) is not None

    def test_promote_best_prefers_earlier_higher_return(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        workspace = prepared_root.current_node
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        workspace.save_checkpoint(
            20,
            {"value": 20},
            metrics={CHECKPOINT_BEST_RETURN_METRIC: 0.9, CHECKPOINT_BEST_LENGTH_METRIC: 12.0},
            force=True,
        )
        workspace.save_checkpoint(
            30,
            {"value": 30},
            metrics={CHECKPOINT_BEST_RETURN_METRIC: 0.4, CHECKPOINT_BEST_LENGTH_METRIC: 40.0},
            force=True,
        )
        prepared_root.save()

        response = promote(prepared_root, PromoteRequest(node_id=workspace.id))

        assert response.checkpoint_step == 20
        assert workspace.resolve_checkpoint_alias(CHECKPOINT_ALIAS_BEST).checkpoint_step == 20


@pytest.mark.parametrize("mutation", ["fork", "extend"])
def test_fork_and_extend_can_start_from_best(
    prepared_root: ExperimentGraph[RLRunConfig],
    mutation: str,
) -> None:
    workspace = prepared_root.current_node
    workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
    workspace.save_checkpoint(
        20,
        {"value": 20},
        metrics={CHECKPOINT_BEST_RETURN_METRIC: 0.9, CHECKPOINT_BEST_LENGTH_METRIC: 12.0},
        force=True,
    )
    workspace.save_checkpoint(
        30,
        {"value": 30},
        metrics={CHECKPOINT_BEST_RETURN_METRIC: 0.4, CHECKPOINT_BEST_LENGTH_METRIC: 40.0},
        force=True,
    )
    promote(prepared_root, PromoteRequest(node_id=workspace.id))
    checkpoint = CheckpointRef(node_id=workspace.id, checkpoint_step=20)

    if mutation == "fork":
        response = fork(
            prepared_root,
            ForkRequest(
                branch="from_best",
                label="child",
                from_checkpoint=checkpoint,
                prepare=True,
            ),
        )
    else:
        response = extend(
            prepared_root,
            ExtendRequest(
                label="child",
                from_checkpoint=checkpoint,
                prepare=True,
            ),
        )

    child = prepared_root.get_node(response.node_id)
    assert child.parent_checkpoint_step == 20
