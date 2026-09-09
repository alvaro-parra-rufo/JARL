"""Tests for the built-in EmptyVariant experiment case."""

from __future__ import annotations

from pathlib import Path

from jarl.envs.navix.custom.empty_variant import EMPTY_VARIANT_ENV_ID
from jarl.envs.navix.scenario_rewards import FLOOR_CELL_SCENARIO_ID, FLOOR_CELL_SCENARIO_VERSION
from jarl.experiments.cases.builtin.navix_empty_variant import CASE
from jarl.experiments.cases.builtin.navix_prepared_root import CASE as PREPARED_EMPTY_CASE
from jarl.experiments.cases.metadata import ExperimentCaseMetadata
from jarl.experiments.io.layout import ExperimentLayout
from jarl.experiments.node import NodeStatus
from jarl.training.config import EnvironmentConfig, RLRunConfig

_CATALOG_FORBIDDEN = (
    "dopamina",
    "dopamine",
    "bonus",
    "trap",
    "cell_entry",
    "floor_cell",
    "occupancy",
    "25.6",
    "16.6",
    "15.93",
    "r_farm",
    "decenas",
    "g=4",
)


class TestNavixEmptyVariantCase:
    def test_materializes_prepared_root_with_overlay_snapshot(self, tmp_path: Path) -> None:
        context = CASE.materialize(tmp_path / "variant")

        graph = context.reload_graph()
        root = graph.get_node(context.node_id("root"))
        config = graph.resolve_config(root)
        snapshot = ExperimentCaseMetadata.load(graph.layout.case_metadata_path)

        assert graph.as_networkx().number_of_nodes() == 1
        assert root.status is NodeStatus.PREPARED
        assert root.branch == "main"
        assert graph.current_node.id == root.id
        assert config.environment.env_id == EMPTY_VARIANT_ENV_ID
        assert config.environment.max_episode_steps is None
        assert config.algorithm.gamma == 0.99
        assert config.environment.reward is not None
        assert config.environment.reward.goal_reached == 1.0
        assert all(
            value == 0.0 for name, value in config.environment.reward.model_dump().items() if name != "goal_reached"
        )
        assert config.environment.scenario_reward_id == FLOOR_CELL_SCENARIO_ID
        assert config.environment.scenario_reward_version == FLOOR_CELL_SCENARIO_VERSION
        assert snapshot.case_id == CASE.spec.id
        assert snapshot.env_id == EMPTY_VARIANT_ENV_ID
        assert snapshot.scenario_reward_id == FLOOR_CELL_SCENARIO_ID
        assert snapshot.scenario_reward_version == FLOOR_CELL_SCENARIO_VERSION

    def test_empty_variant_without_overlay_is_not_this_case(self) -> None:
        config = RLRunConfig(environment=EnvironmentConfig(env_id=EMPTY_VARIANT_ENV_ID))

        assert config.environment.reward is None
        assert config.environment.scenario_reward_id is None
        assert config.environment.scenario_reward_version is None

    def test_prepared_empty_root_does_not_write_this_snapshot(self, tmp_path: Path) -> None:
        context = PREPARED_EMPTY_CASE.materialize(tmp_path / "empty")

        graph = context.reload_graph()
        config = graph.resolve_config(graph.current_node)

        assert config.environment.env_id != EMPTY_VARIANT_ENV_ID
        assert config.environment.scenario_reward_id is None
        assert not ExperimentLayout(context.experiment_dir).case_metadata_path.exists()

    def test_catalog_text_hides_overlay_and_tabular_thresholds(self) -> None:
        surfaces = " ".join(
            [
                CASE.spec.id,
                CASE.spec.title,
                CASE.spec.description,
                *CASE.spec.tags,
            ]
        ).lower()

        for token in _CATALOG_FORBIDDEN:
            assert token not in surfaces
