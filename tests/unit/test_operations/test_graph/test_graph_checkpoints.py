"""Tests for graph checkpoint overview and search operations."""

from __future__ import annotations

import pytest

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import (
    CHECKPOINT_BEST_LENGTH_METRIC,
    CHECKPOINT_BEST_RETURN_METRIC,
)
from jarl.operations.contracts.constants import CHECKPOINTS_MAX_LIMIT
from jarl.operations.graph.checkpoints import CheckpointsRequest, checkpoints
from jarl.operations.graph.promote import PromoteRequest, promote
from jarl.training.config import RLRunConfig


def _seed_diverging_checkpoints(graph: ExperimentGraph[RLRunConfig]) -> str:
    workspace = graph.current_node
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
    promote(graph, PromoteRequest(node_id=workspace.id))
    return workspace.id


class TestCheckpointsOperation:
    def test_overview_returns_aliases_without_candidates(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        node_id = _seed_diverging_checkpoints(prepared_root)

        response = checkpoints(prepared_root, CheckpointsRequest(node_id=node_id))

        assert response.mode == "overview"
        assert response.checkpoint_count == 2
        assert response.best is not None
        assert response.best.checkpoint_step == 20
        assert response.best.eval_return == 0.9
        assert response.best.eval_length == 12.0
        assert "best" in response.best.aliases
        assert response.latest is not None
        assert response.latest.checkpoint_step == 30
        assert response.candidates == ()

    def test_search_sorts_by_eval_return_and_limits(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        node_id = _seed_diverging_checkpoints(prepared_root)

        response = checkpoints(
            prepared_root,
            CheckpointsRequest(
                node_id=node_id,
                sort_by=CHECKPOINT_BEST_RETURN_METRIC,
                sort_descending=True,
                limit=1,
            ),
        )

        assert response.mode == "search"
        assert response.returned_count == 1
        assert response.has_more is True
        assert response.candidates[0].checkpoint_step == 20
        assert response.best is not None
        assert response.best.checkpoint_step == 20

    def test_search_filters_step_range(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        node_id = _seed_diverging_checkpoints(prepared_root)

        response = checkpoints(
            prepared_root,
            CheckpointsRequest(node_id=node_id, step_min=25, step_max=35, limit=10),
        )

        assert [item.checkpoint_step for item in response.candidates] == [30]

    def test_rejects_inverted_step_range(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        node_id = prepared_root.current_node.id

        with pytest.raises(ValueError, match="step_min"):
            checkpoints(
                prepared_root,
                CheckpointsRequest(node_id=node_id, step_min=40, step_max=10),
            )

    def test_rejects_limit_above_cap(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        node_id = prepared_root.current_node.id

        with pytest.raises(ValueError, match="limit"):
            checkpoints(
                prepared_root,
                CheckpointsRequest(node_id=node_id, limit=CHECKPOINTS_MAX_LIMIT + 1),
            )
