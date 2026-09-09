"""Tests for the graph reward read operation."""

from __future__ import annotations

from jarl.envs.navix.custom.empty_variant import EMPTY_VARIANT_ENV_ID
from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.envs.navix.scenario_rewards import FLOOR_CELL_SCENARIO_ID, FLOOR_CELL_SCENARIO_VERSION
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeStatus
from jarl.experiments.summaries import config_highlights
from jarl.operations.graph.reward import RewardRequest, reward
from jarl.operations.graph.set_reward import SetRewardRequest, set_reward
from jarl.training.config import RLRunConfig

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


class TestRewardRead:
    def test_native_mix_is_null(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        response = reward(prepared_root, RewardRequest())

        compact = response.to_compact_dict()

        assert compact == {"node_id": prepared_root.current_node.id, "reward": None}

    def test_custom_mix_returns_catalog_keys_only(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        set_reward(prepared_root, SetRewardRequest(weights={"goal_reached": 4.0}))

        compact = reward(prepared_root, RewardRequest()).to_compact_dict()
        dumped = str(compact).lower()

        assert compact["reward"] is not None
        assert tuple(compact["reward"]) == _CHANNEL_NAMES
        assert compact["reward"]["goal_reached"] == 4.0
        assert compact["reward"]["key_pickup"] == 0.0
        for token in _LEAKED_SURFACE:
            assert token not in dumped

    def test_read_ok_on_completed_node(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        set_reward(prepared_root, SetRewardRequest(weights={"goal_reached": 4.0}))
        prepared_root.current_node._meta.status = NodeStatus.COMPLETED

        compact = reward(prepared_root, RewardRequest()).to_compact_dict()

        assert compact["reward"] is not None
        assert compact["reward"]["goal_reached"] == 4.0

    def test_read_hides_overlay_and_highlights_stay_clean(
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

        compact = reward(empty_graph, RewardRequest()).to_compact_dict()
        highlights = config_highlights(empty_graph.resolve_config(empty_graph.current_node))
        dumped = f"{compact}{highlights}".lower()

        assert compact["reward"] is not None
        assert tuple(compact["reward"]) == _CHANNEL_NAMES
        assert "reward" not in highlights
        for token in _LEAKED_SURFACE:
            assert token not in dumped
        assert "cell_entry" not in compact["reward"]
