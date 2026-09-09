"""Tests for scenario reward overlay lookup."""

from __future__ import annotations

import pytest

from jarl.envs.navix.custom import CELL_ENTRY_POSITION
from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.envs.navix.scenario_rewards import (
    EMPTY_VARIANT_ENV_ID,
    FLOOR_CELL_SCENARIO_ID,
    FLOOR_CELL_SCENARIO_VERSION,
    ScenarioRewardError,
    resolve_scenario_reward_spec,
    scenario_reward_fn,
)
from jarl.training.config import AlgorithmConfig, EnvironmentConfig, RLRunConfig


class TestResolveScenarioRewardSpec:
    """Registry lookup is explicit and fails before the environment is built."""

    def test_absent_overlay_returns_none(self) -> None:
        spec = resolve_scenario_reward_spec("Navix-Empty-5x5-v0", None, None)

        assert spec is None

    def test_empty_variant_v1_is_compatible(self) -> None:
        spec = resolve_scenario_reward_spec(
            EMPTY_VARIANT_ENV_ID,
            FLOOR_CELL_SCENARIO_ID,
            FLOOR_CELL_SCENARIO_VERSION,
        )

        assert spec is not None
        assert spec.id == FLOOR_CELL_SCENARIO_ID
        assert spec.version == FLOOR_CELL_SCENARIO_VERSION
        assert spec.scale == 0.5
        assert spec.position == CELL_ENTRY_POSITION
        assert EMPTY_VARIANT_ENV_ID in spec.compatible_env_ids

    def test_incompatible_env_id_raises_detailed_error(self) -> None:
        with pytest.raises(ScenarioRewardError, match="incompatible with env_id") as exc_info:
            resolve_scenario_reward_spec(
                "Navix-Empty-5x5-v0",
                FLOOR_CELL_SCENARIO_ID,
                FLOOR_CELL_SCENARIO_VERSION,
            )

        message = str(exc_info.value)
        assert FLOOR_CELL_SCENARIO_ID in message
        assert "Navix-Empty-5x5-v0" in message

    def test_unknown_key_raises_detailed_error(self) -> None:
        with pytest.raises(ScenarioRewardError, match="Unknown ScenarioRewardSpec"):
            resolve_scenario_reward_spec(EMPTY_VARIANT_ENV_ID, "navix.missing", 1)

    def test_xor_id_version_raises(self) -> None:
        with pytest.raises(ScenarioRewardError, match="both be set"):
            resolve_scenario_reward_spec(EMPTY_VARIANT_ENV_ID, FLOOR_CELL_SCENARIO_ID, None)

    def test_scenario_reward_fn_is_scaled_occupancy(self) -> None:
        spec = resolve_scenario_reward_spec(
            EMPTY_VARIANT_ENV_ID,
            FLOOR_CELL_SCENARIO_ID,
            FLOOR_CELL_SCENARIO_VERSION,
        )
        assert spec is not None
        fn = scenario_reward_fn(spec)
        assert callable(fn)


class TestEnvironmentOverlayInvariants:
    """Config JSON cannot enable an overlay without a custom mix."""

    def test_overlay_requires_reward_block(self) -> None:
        with pytest.raises(ValueError, match=r"requires environment\.reward"):
            EnvironmentConfig(
                scenario_reward_id=FLOOR_CELL_SCENARIO_ID,
                scenario_reward_version=FLOOR_CELL_SCENARIO_VERSION,
            )

    def test_id_xor_version_rejected(self) -> None:
        with pytest.raises(ValueError, match="both be set"):
            EnvironmentConfig(
                reward=RewardWeightsConfig(),
                scenario_reward_id=FLOOR_CELL_SCENARIO_ID,
            )
        with pytest.raises(ValueError, match="both be set"):
            EnvironmentConfig(
                reward=RewardWeightsConfig(),
                scenario_reward_version=FLOOR_CELL_SCENARIO_VERSION,
            )

    def test_paired_overlay_with_mix_is_valid(self) -> None:
        config = EnvironmentConfig(
            env_id=EMPTY_VARIANT_ENV_ID,
            reward=RewardWeightsConfig(),
            scenario_reward_id=FLOOR_CELL_SCENARIO_ID,
            scenario_reward_version=FLOOR_CELL_SCENARIO_VERSION,
        )

        assert config.reward is not None
        assert config.scenario_reward_id == FLOOR_CELL_SCENARIO_ID

    def test_both_none_is_valid(self) -> None:
        config = EnvironmentConfig()

        assert config.reward is None
        assert config.scenario_reward_id is None
        assert config.scenario_reward_version is None

    def test_fork_without_map_change_inherits_overlay(self) -> None:
        parent = RLRunConfig(
            environment=EnvironmentConfig(
                env_id=EMPTY_VARIANT_ENV_ID,
                nr_envs=8,
                reward=RewardWeightsConfig(),
                scenario_reward_id=FLOOR_CELL_SCENARIO_ID,
                scenario_reward_version=FLOOR_CELL_SCENARIO_VERSION,
            ),
            algorithm=AlgorithmConfig(nr_steps=128, minibatch_size=1024),
        )
        child = parent.apply_overrides({"algorithm.learning_rate": 1e-4})

        assert child.environment.scenario_reward_id == parent.environment.scenario_reward_id
        assert child.environment.scenario_reward_version == parent.environment.scenario_reward_version
        spec = resolve_scenario_reward_spec(
            child.environment.env_id,
            child.environment.scenario_reward_id,
            child.environment.scenario_reward_version,
        )
        assert spec is not None

    def test_fork_to_incompatible_map_fails_at_resolve(self) -> None:
        parent = RLRunConfig(
            environment=EnvironmentConfig(
                env_id=EMPTY_VARIANT_ENV_ID,
                nr_envs=8,
                reward=RewardWeightsConfig(),
                scenario_reward_id=FLOOR_CELL_SCENARIO_ID,
                scenario_reward_version=FLOOR_CELL_SCENARIO_VERSION,
            ),
            algorithm=AlgorithmConfig(nr_steps=128, minibatch_size=1024),
        )
        child = parent.apply_overrides({"environment.env_id": "Navix-Empty-5x5-v0"})

        assert child.environment.scenario_reward_id == FLOOR_CELL_SCENARIO_ID
        with pytest.raises(ScenarioRewardError, match="incompatible"):
            resolve_scenario_reward_spec(
                child.environment.env_id,
                child.environment.scenario_reward_id,
                child.environment.scenario_reward_version,
            )
