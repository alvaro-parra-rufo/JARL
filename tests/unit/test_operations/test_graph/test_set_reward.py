"""Tests for the graph set-reward mutation operation."""

from __future__ import annotations

import math

import pytest

from jarl.envs.navix.custom.empty_variant import EMPTY_VARIANT_ENV_ID
from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.envs.navix.scenario_rewards import FLOOR_CELL_SCENARIO_ID, FLOOR_CELL_SCENARIO_VERSION
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeStatus
from jarl.experiments.summaries import config_highlights
from jarl.operations.graph.fork import ForkRequest, fork
from jarl.operations.graph.reward import RewardRequest, reward
from jarl.operations.graph.set_reward import SetRewardRequest, set_reward
from jarl.training.config import RLRunConfig
from jarl.training.override_models import ForkConfigOverrides, RetrainConfigOverrides

_CHANNEL_NAMES = tuple(RewardWeightsConfig.model_fields)
_LEAKED_SURFACE = (
    "dopamina",
    "bonus",
    "trap",
    "cell_entry",
    "floor_cell",
    "compatible_env_ids",
    "scenario_reward",
    "25.6",
    "16.6",
    "15.93",
    "r_farm",
)


class TestSetReward:
    def test_native_none_starts_from_task_default_and_patches(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        compact = set_reward(prepared_root, SetRewardRequest(weights={"goal_reached": 4.0})).to_compact_dict()

        mix = prepared_root.resolve_config(prepared_root.current_node).environment.reward

        assert mix is not None
        assert mix.goal_reached == 4.0
        assert mix.key_pickup == 0.0
        assert tuple(compact["reward"]) == _CHANNEL_NAMES
        assert compact["reward"]["goal_reached"] == 4.0
        assert compact["node_id"] == prepared_root.current_node.id
        assert compact["status"] == "prepared"

    def test_sparse_merge_keeps_unspecified_channels(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        set_reward(prepared_root, SetRewardRequest(weights={"goal_reached": 4.0, "goal_approach": 2.0}))

        compact = set_reward(prepared_root, SetRewardRequest(weights={"goal_approach": 1.1})).to_compact_dict()
        mix = prepared_root.resolve_config(prepared_root.current_node).environment.reward

        assert mix is not None
        assert mix.goal_reached == 4.0
        assert mix.goal_approach == 1.1
        assert compact["reward"]["goal_reached"] == 4.0
        assert compact["reward"]["goal_approach"] == 1.1

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(4.0, id="four"),
            pytest.param(1.1, id="slightly-above-one"),
        ],
    )
    def test_accepts_weights_above_one(self, prepared_root: ExperimentGraph[RLRunConfig], value: float) -> None:
        compact = set_reward(prepared_root, SetRewardRequest(weights={"goal_reached": value})).to_compact_dict()

        assert compact["reward"]["goal_reached"] == value

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(-1.0, id="negative"),
            pytest.param(math.nan, id="nan"),
            pytest.param(math.inf, id="inf"),
        ],
    )
    def test_rejects_invalid_weights(self, prepared_root: ExperimentGraph[RLRunConfig], value: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            set_reward(prepared_root, SetRewardRequest(weights={"goal_reached": value}))

    def test_rejects_unknown_and_cell_entry_channels(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        with pytest.raises(ValueError, match="Unknown reward channels"):
            set_reward(prepared_root, SetRewardRequest(weights={"cell_entry": 0.5}))

    def test_rejects_empty_patch(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        with pytest.raises(ValueError, match="at least one"):
            set_reward(prepared_root, SetRewardRequest(weights={}))

    def test_rejects_completed_node_but_read_still_works(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        set_reward(prepared_root, SetRewardRequest(weights={"goal_reached": 4.0}))
        prepared_root.current_node._meta.status = NodeStatus.COMPLETED

        with pytest.raises(ValueError, match="already trained"):
            set_reward(prepared_root, SetRewardRequest(weights={"goal_reached": 8.0}))

        compact = reward(prepared_root, RewardRequest()).to_compact_dict()
        assert compact["reward"] is not None
        assert compact["reward"]["goal_reached"] == 4.0

    def test_rejects_node_with_checkpoints(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        workspace = prepared_root.current_node
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        workspace.save_checkpoint(1, {"value": 1}, force=True)

        with pytest.raises(ValueError, match="checkpoints"):
            set_reward(prepared_root, SetRewardRequest(weights={"goal_reached": 4.0}))

        assert reward(prepared_root, RewardRequest()).to_compact_dict()["reward"] is None

    def test_persists_in_place_and_reloads(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        node_id = prepared_root.current_node.id
        set_reward(prepared_root, SetRewardRequest(weights={"goal_reached": 4.0}))

        reloaded = ExperimentGraph.from_directory(prepared_root.layout.root, config_cls=RLRunConfig)
        mix = reloaded.resolve_config(reloaded.get_node(node_id)).environment.reward

        assert mix is not None
        assert mix.goal_reached == 4.0
        assert reloaded.get_node(node_id).id == node_id

    def test_preserves_overlay_and_hides_it_from_compact(
        self,
        empty_graph: ExperimentGraph[RLRunConfig],
    ) -> None:
        config = RLRunConfig().apply_overrides(
            {
                "environment.env_id": EMPTY_VARIANT_ENV_ID,
                "environment.reward": RewardWeightsConfig(goal_reached=1.0).model_dump(),
                "environment.scenario_reward_id": FLOOR_CELL_SCENARIO_ID,
                "environment.scenario_reward_version": FLOOR_CELL_SCENARIO_VERSION,
            }
        )
        empty_graph.create_root(config, branch="main", label="overlay", prepare=True)
        empty_graph.save()

        compact = set_reward(empty_graph, SetRewardRequest(weights={"goal_reached": 4.0})).to_compact_dict()
        resolved = empty_graph.resolve_config(empty_graph.current_node)
        highlights = config_highlights(resolved)
        dumped = f"{compact}{highlights}".lower()

        assert resolved.environment.scenario_reward_id == FLOOR_CELL_SCENARIO_ID
        assert resolved.environment.scenario_reward_version == FLOOR_CELL_SCENARIO_VERSION
        assert resolved.environment.reward is not None
        assert resolved.environment.reward.goal_reached == 4.0
        assert "scenario_reward" not in compact
        assert "reward" not in highlights
        for token in _LEAKED_SURFACE:
            assert token not in dumped

    def test_fork_and_retrain_schemas_still_omit_reward_paths(self) -> None:
        for model in (ForkConfigOverrides, RetrainConfigOverrides):
            properties = model.model_json_schema()["properties"]
            assert "environment.reward.goal_reached" not in properties
            assert "environment.scenario_reward_id" not in properties
            assert "environment.scenario_reward_version" not in properties

    def test_does_not_create_a_child_node(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        node_id = prepared_root.current_node.id

        set_reward(prepared_root, SetRewardRequest(weights={"goal_reached": 4.0}))

        assert prepared_root.current_node.id == node_id
        assert prepared_root.as_networkx().number_of_nodes() == 1

    def test_child_inherits_parent_mix_until_patched(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        parent_id = prepared_root.current_node.id
        set_reward(prepared_root, SetRewardRequest(weights={"goal_reached": 4.0}))
        child = fork(prepared_root, ForkRequest(branch="exp", label="child", prepare=True))

        child_mix = prepared_root.resolve_config(prepared_root.get_node(child.node_id)).environment.reward
        set_reward(
            prepared_root,
            SetRewardRequest(node_id=child.node_id, weights={"distance_to_goal": 1.1}),
        )
        patched = prepared_root.resolve_config(prepared_root.get_node(child.node_id)).environment.reward
        parent_mix = prepared_root.resolve_config(prepared_root.get_node(parent_id)).environment.reward

        assert child_mix is not None
        assert child_mix.goal_reached == 4.0
        assert patched is not None
        assert patched.goal_reached == 4.0
        assert patched.distance_to_goal == 1.1
        assert parent_mix is not None
        assert parent_mix.distance_to_goal == 0.0
