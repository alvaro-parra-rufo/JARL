"""Tests for extend semantics anchored to branch heads."""

from __future__ import annotations

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import CheckpointRef
from jarl.operations.graph.checkout import CheckoutRequest, checkout
from jarl.operations.graph.extend import ExtendRequest, extend
from jarl.operations.graph.fork import ForkRequest, fork
from jarl.training.config import RLRunConfig


class TestExtendUsesBranchHead:
    def test_extend_branch_uses_head_even_when_current_node_elsewhere(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        root_id = prepared_root.current_node.id
        first = fork(prepared_root, ForkRequest(branch="exp", label="first", prepare=True))
        second = extend(prepared_root, ExtendRequest(branch="exp", label="second", prepare=True))
        checkout(prepared_root, CheckoutRequest(node_id=root_id))

        third = extend(prepared_root, ExtendRequest(branch="exp", label="third", prepare=True))

        child = prepared_root.get_node(third.node_id)

        assert third.parent_id == second.node_id
        assert child.node_metadata.parent_id == second.node_id
        assert first.node_id != third.parent_id

    def test_extend_from_checkpoint_anchors_to_branch_head(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        first = fork(prepared_root, ForkRequest(branch="exp", label="first", prepare=True))
        workspace = prepared_root.get_node(first.node_id)
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        workspace.save_checkpoint(1, {"value": 1}, force=True)
        prepared_root.save()

        second = extend(
            prepared_root,
            ExtendRequest(
                branch="exp",
                label="second",
                from_checkpoint=CheckpointRef(node_id=first.node_id, checkpoint_step=1),
                prepare=True,
            ),
        )

        assert prepared_root.head("exp").id == second.node_id
        assert second.parent_id == first.node_id
