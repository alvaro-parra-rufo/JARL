"""Tests for graph mutation operations."""

from __future__ import annotations

from dataclasses import replace

import pytest

from jarl.experiments.graph import ExperimentGraph
from jarl.operations.graph.checkout import CheckoutRequest, checkout
from jarl.operations.graph.create_root import CreateRootRequest, create_root
from jarl.operations.graph.fork import ForkRequest, fork
from jarl.training.config import RLRunConfig
from jarl.training.presets import RunFormPayload, default_form_values


class TestCreateRoot:
    def test_create_root_prepared(self, empty_graph: ExperimentGraph[RLRunConfig]) -> None:
        response = create_root(
            empty_graph,
            CreateRootRequest(label="baseline", branch="main"),
        )

        assert response.status == "prepared"
        assert response.branch == "main"
        assert empty_graph.current_node.id == response.node_id

    def test_create_root_rejects_existing_nodes(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        with pytest.raises(ValueError, match="already has"):
            create_root(prepared_root, CreateRootRequest(label="duplicate"))

    def test_create_root_custom_preset_uses_production_defaults(
        self, empty_graph: ExperimentGraph[RLRunConfig]
    ) -> None:
        response = create_root(
            empty_graph,
            CreateRootRequest(label="baseline", preset="custom"),
        )

        config = empty_graph.resolve_config(empty_graph.get_node(response.node_id))
        assert config.algorithm.total_timesteps == 5_000_000
        assert config.environment.max_episode_steps is None

    def test_create_root_form_sets_max_episode_steps(self, empty_graph: ExperimentGraph[RLRunConfig]) -> None:
        values = replace(default_form_values(preset="custom"), max_episode_steps=100)
        response = create_root(
            empty_graph,
            CreateRootRequest(
                label="horizon",
                preset="custom",
                form=RunFormPayload(values=values, video_frequency=0, record_final_video=True),
            ),
        )

        config = empty_graph.resolve_config(empty_graph.get_node(response.node_id))
        assert config.environment.max_episode_steps == 100


class TestCheckout:
    def test_checkout_moves_current_node(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        child = prepared_root.fork("exp", from_node=prepared_root.current_node, prepare=True)
        prepared_root.save()

        response = checkout(prepared_root, CheckoutRequest(node_id=child.id))

        assert response.node_id == child.id
        assert prepared_root.current_node.id == child.id


class TestForkExtend:
    def test_fork_creates_child(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        parent_id = prepared_root.current_node.id

        response = fork(
            prepared_root,
            ForkRequest(branch="exp", label="child", prepare=True),
        )

        assert response.parent_id == parent_id
        assert response.branch == "exp"
        assert prepared_root.get_node(response.node_id).node_metadata.parent_id == parent_id

    def test_fork_applies_max_episode_steps_override(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        response = fork(
            prepared_root,
            ForkRequest(
                branch="exp",
                label="horizon",
                prepare=True,
                config_overrides={"environment.max_episode_steps": 100},
            ),
        )

        config = prepared_root.resolve_config(prepared_root.get_node(response.node_id))
        assert config.environment.max_episode_steps == 100

    def test_fork_rejects_immutable_config_override(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        with pytest.raises(ValueError, match="immutable paths"):
            fork(
                prepared_root,
                ForkRequest(
                    branch="exp",
                    label="child",
                    prepare=True,
                    config_overrides={"algorithm.name": "ppo.full_jax.navix"},
                ),
            )
