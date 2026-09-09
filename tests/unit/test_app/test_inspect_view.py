"""Tests for inspection tables used by Runner Lab."""

from __future__ import annotations

from pathlib import Path

from jarl.app.lib.inspect_view import reward_mix_table, scenario_overlay_table
from jarl.envs.navix.custom.empty_variant import EMPTY_VARIANT_ENV_ID
from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.envs.navix.scenario_rewards import FLOOR_CELL_SCENARIO_ID, FLOOR_CELL_SCENARIO_VERSION
from jarl.experiments.graph import ExperimentGraph
from jarl.operations.graph.set_reward import SetRewardRequest, set_reward
from jarl.training.config import RLRunConfig


class TestRewardMixTable:
    """Resolved Navix reward mix shown in inspection panels."""

    def test_native_mix_is_empty(self, tmp_path: Path) -> None:
        graph = ExperimentGraph(tmp_path / "exp", base_config=RLRunConfig())
        graph.create_root(RLRunConfig(), branch="main", label="baseline", prepare=True)

        table = reward_mix_table(graph, graph.current_node)

        assert list(table.columns) == ["channel", "weight", "description"]
        assert table.empty

    def test_custom_mix_lists_channels_heaviest_first(self, tmp_path: Path) -> None:
        graph = ExperimentGraph(tmp_path / "exp", base_config=RLRunConfig())
        graph.create_root(RLRunConfig(), branch="main", label="baseline", prepare=True)
        set_reward(graph, SetRewardRequest(weights={"goal_reached": 4.0, "key_pickup": 1.5}))

        table = reward_mix_table(graph, graph.current_node)

        assert table.iloc[0]["channel"] == "goal_reached"
        assert table.iloc[0]["weight"] == 4.0
        assert table.iloc[1]["channel"] == "key_pickup"
        assert table.iloc[1]["weight"] == 1.5
        assert "Pulse when the goal-reached event fires" in str(table.iloc[0]["description"])


class TestScenarioOverlayTable:
    """Scenario overlay rows for maps that define one."""

    def test_absent_overlay_is_empty(self) -> None:
        table = scenario_overlay_table(RLRunConfig())

        assert table.empty

    def test_floor_cell_overlay_rows(self) -> None:
        config = RLRunConfig().apply_overrides(
            {
                "environment.env_id": EMPTY_VARIANT_ENV_ID,
                "environment.reward": RewardWeightsConfig(goal_reached=1.0).model_dump(),
                "environment.scenario_reward_id": FLOOR_CELL_SCENARIO_ID,
                "environment.scenario_reward_version": FLOOR_CELL_SCENARIO_VERSION,
            }
        )

        table = scenario_overlay_table(config)

        assert len(table) == 1
        assert table.iloc[0]["id"] == FLOOR_CELL_SCENARIO_ID
        assert int(table.iloc[0]["version"]) == FLOOR_CELL_SCENARIO_VERSION
        assert table.iloc[0]["scale"] == 0.5
