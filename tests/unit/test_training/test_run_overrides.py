"""Tests for ``jarl.training.run_overrides``."""

from __future__ import annotations

import pytest

from jarl.training.config import RLRunConfig
from jarl.training.run_overrides import (
    ensure_validated_config_overrides,
    merge_validated_config_overrides,
    mutable_overrides_from_config,
    validate_config_overrides,
    validate_fork_overrides,
    validate_retrain_overrides,
)
from tests.helpers.minimal_navix_configs import minimal_ppo_run_config


@pytest.fixture()
def base_config() -> RLRunConfig:
    """Resolved parent-style config used as merge base."""
    return minimal_ppo_run_config()


class TestMutableOverridesFromConfig:
    def test_paths_match_retrain_mutable_paths(self, base_config: RLRunConfig) -> None:
        overrides = mutable_overrides_from_config(base_config)

        assert set(overrides) == set(RLRunConfig.RETRAIN_MUTABLE_PATHS)

    def test_excludes_environment_fields(self, base_config: RLRunConfig) -> None:
        overrides = mutable_overrides_from_config(base_config)

        assert "environment.env_id" not in overrides
        assert "environment.nr_envs" not in overrides


class TestValidateConfigOverrides:
    def test_retrain_accepts_mutable_overrides(self, base_config: RLRunConfig) -> None:
        validate_retrain_overrides(mutable_overrides_from_config(base_config))

    def test_retrain_rejects_immutable_paths(self) -> None:
        with pytest.raises(ValueError, match="immutable paths"):
            validate_config_overrides({"algorithm.name": "ppo.full_jax.navix"}, policy="retrain")

    def test_retrain_rejects_unknown_paths(self) -> None:
        with pytest.raises(ValueError, match="RETRAIN_MUTABLE_PATHS"):
            validate_retrain_overrides({"environment.env_id": "Navix-Empty-5x5-v0"})

    def test_fork_allows_environment_env_id(self, base_config: RLRunConfig) -> None:
        overrides = mutable_overrides_from_config(base_config)
        overrides["environment.env_id"] = "Navix-Empty-5x5-v0"

        validate_fork_overrides(overrides)

    def test_fork_allows_environment_max_episode_steps(self, base_config: RLRunConfig) -> None:
        overrides = mutable_overrides_from_config(base_config)
        overrides["environment.max_episode_steps"] = 100

        validate_fork_overrides(overrides)

    def test_fork_rejects_other_environment_fields(self, base_config: RLRunConfig) -> None:
        overrides = mutable_overrides_from_config(base_config)
        overrides["environment.nr_envs"] = 8

        with pytest.raises(ValueError, match="fork/extend"):
            validate_fork_overrides(overrides)

    @pytest.mark.parametrize(
        "path",
        [
            pytest.param("environment.reward.goal_reached", id="reward-weight"),
            pytest.param("environment.scenario_reward_id", id="scenario-id"),
            pytest.param("environment.scenario_reward_version", id="scenario-version"),
        ],
    )
    def test_fork_and_retrain_reject_reward_and_overlay_paths(self, path: str) -> None:
        with pytest.raises(ValueError, match="RETRAIN_MUTABLE_PATHS"):
            validate_retrain_overrides({path: 1.0 if "reward" in path else "navix.floor_cell"})
        with pytest.raises(ValueError, match="fork/extend"):
            validate_fork_overrides({path: 1.0 if "version" not in path else 1})

    def test_retrain_rejects_highlight_shorthand_keys(self) -> None:
        with pytest.raises(ValueError, match=r"algorithm\.evaluation_and_save_frequency"):
            validate_retrain_overrides(
                {
                    "algorithm.eval_frequency": 256,
                    "algorithm.nr_envs": 16,
                }
            )


class TestMergeValidatedConfigOverrides:
    def test_returns_none_for_empty_overrides(self, base_config: RLRunConfig) -> None:
        assert merge_validated_config_overrides(base_config, None, policy="retrain") is None
        assert merge_validated_config_overrides(base_config, {}, policy="fork") is None

    def test_returns_merged_config(self, base_config: RLRunConfig) -> None:
        merged = merge_validated_config_overrides(
            base_config,
            {"algorithm.learning_rate": 1e-5},
            policy="retrain",
        )

        assert merged is not None
        assert merged.algorithm.learning_rate == 1e-5
        assert merged.environment.nr_envs == base_config.environment.nr_envs

    def test_validates_rollout_batch_on_final_config(self, base_config: RLRunConfig) -> None:
        with pytest.raises(ValueError, match="minibatch_size"):
            merge_validated_config_overrides(
                base_config,
                {"algorithm.nr_steps": 7},
                policy="retrain",
            )

    def test_allows_learning_rate_only_override(self, base_config: RLRunConfig) -> None:
        merged = merge_validated_config_overrides(
            base_config,
            {"algorithm.learning_rate": 2e-4},
            policy="retrain",
        )

        assert merged is not None
        assert merged.algorithm.nr_steps == base_config.algorithm.nr_steps

    def test_fork_policy_uses_parent_nr_envs_for_batch(self, base_config: RLRunConfig) -> None:
        with pytest.raises(ValueError, match="minibatch_size"):
            merge_validated_config_overrides(
                base_config,
                {"algorithm.nr_steps": 7, "algorithm.minibatch_size": 32},
                policy="fork",
            )


class TestEnsureValidatedConfigOverrides:
    def test_accepts_valid_sparse_overrides(self, base_config: RLRunConfig) -> None:
        ensure_validated_config_overrides(
            base_config,
            {"algorithm.learning_rate": 2e-4},
            policy="retrain",
        )

    def test_rejects_invalid_sparse_overrides(self, base_config: RLRunConfig) -> None:
        with pytest.raises(ValueError, match="immutable paths"):
            ensure_validated_config_overrides(
                base_config,
                {"algorithm.name": "ppo.full_jax.navix"},
                policy="retrain",
            )

    def test_noop_for_empty_overrides(self, base_config: RLRunConfig) -> None:
        ensure_validated_config_overrides(base_config, None, policy="retrain")
