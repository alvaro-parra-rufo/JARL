"""Tests for RL training configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jarl.config import ConfigDiff
from jarl.experiments.graph import ExperimentGraph
from jarl.training.config import AlgorithmConfig, EnvironmentConfig, RewardWeightsConfig, RLRunConfig


@pytest.fixture()
def rl_config() -> RLRunConfig:
    """Small RL config suitable for unit tests."""
    return RLRunConfig(
        environment=EnvironmentConfig(nr_envs=8, seed=7),
        algorithm=AlgorithmConfig(
            total_timesteps=8192,
            nr_steps=128,
            minibatch_size=1024,
            evaluation_and_save_frequency=-1,
        ),
    )


class TestRLRunConfigDefaults:
    """Tests for default RL config construction."""

    def test_applies_run_config_defaults(self) -> None:
        config = RLRunConfig()

        assert config.project_name == "navix"
        assert config.exp_name == "ppo_full_jax"
        assert config.run_name
        assert config.environment.env_id == "Navix-LavaGapS5-v0"
        assert config.algorithm.name == "ppo.full_jax.navix"
        assert config.algorithm.requested_total_timesteps is None
        assert config.runner.save_model is True
        assert config.video.record_video is True
        assert config.environment.reward is None
        assert config.environment.scenario_reward_id is None

    def test_derived_schedule_fields_default_to_none(self) -> None:
        config = RLRunConfig()

        assert config.algorithm.batch_size is None
        assert config.algorithm.actual_total_timesteps is None
        assert config.algorithm.effective_evaluation_and_save_frequency is None


class TestEnvironmentOverlayInvariants:
    """Overlay fields are paired and require a custom reward mix."""

    def test_overlay_requires_reward_mix(self) -> None:
        with pytest.raises(ValidationError, match=r"requires environment\.reward"):
            EnvironmentConfig(scenario_reward_id="navix.floor_cell", scenario_reward_version=1)

    def test_mix_and_overlay_round_trip(self, rl_config: RLRunConfig, tmp_path: Path) -> None:
        updated = rl_config.apply_overrides(
            {
                "environment.reward": RewardWeightsConfig(goal_reached=1.0).model_dump(),
                "environment.scenario_reward_id": "navix.floor_cell",
                "environment.scenario_reward_version": 1,
            }
        )
        path = updated.save(tmp_path / "config.json")
        loaded = RLRunConfig.load(path)

        assert loaded.environment.reward is not None
        assert loaded.environment.reward.goal_reached == 1.0
        assert loaded.environment.scenario_reward_id == "navix.floor_cell"
        assert loaded.environment.scenario_reward_version == 1


class TestRLRunConfigValidation:
    """Tests for RL config validators."""

    def test_rejects_minibatch_that_does_not_divide_rollout_batch(self) -> None:
        with pytest.raises(ValidationError, match="minibatch_size"):
            RLRunConfig(
                environment=EnvironmentConfig(nr_envs=8),
                algorithm=AlgorithmConfig(nr_steps=128, minibatch_size=1000),
            )

    @pytest.mark.parametrize(
        ("evaluation_and_save_frequency",),
        [
            pytest.param(0, id="zero"),
        ],
    )
    def test_rejects_invalid_evaluation_frequency(self, evaluation_and_save_frequency: int) -> None:
        with pytest.raises(ValidationError, match="evaluation_and_save_frequency"):
            RLRunConfig(
                algorithm=AlgorithmConfig(evaluation_and_save_frequency=evaluation_and_save_frequency),
            )

    def test_accepts_auto_evaluation_frequency(self) -> None:
        config = RLRunConfig(algorithm=AlgorithmConfig(evaluation_and_save_frequency=-1))

        assert config.algorithm.evaluation_and_save_frequency == -1


class TestRLRunConfigOverrides:
    """Tests for diffing and override application on nested RL configs."""

    def test_apply_overrides_with_dotted_nested_keys(self, rl_config: RLRunConfig) -> None:
        updated = rl_config.apply_overrides(
            {
                "algorithm.learning_rate": 1e-4,
                "environment.nr_envs": 16,
                "video.record_video": False,
            }
        )

        assert updated.algorithm.learning_rate == 1e-4
        assert updated.environment.nr_envs == 16
        assert updated.video.record_video is False
        assert updated.algorithm.total_timesteps == rl_config.algorithm.total_timesteps

    def test_diff_reports_nested_algorithm_changes(self, rl_config: RLRunConfig) -> None:
        child = rl_config.apply_overrides({"algorithm.total_timesteps": 4096})

        diff = rl_config.diff(child)

        assert diff == ConfigDiff(
            added={},
            removed={},
            changed={"algorithm": (rl_config.algorithm.model_dump(), child.algorithm.model_dump())},
        )

    def test_resolve_config_overrides_returns_sparse_nested_diff(self, rl_config: RLRunConfig) -> None:
        child = rl_config.apply_overrides(
            {
                "algorithm.total_timesteps": 4096,
                "environment.seed": 99,
            }
        )

        overrides = rl_config.resolve_config_overrides(target=child, flatten=True)

        assert overrides == {
            "algorithm.total_timesteps": 4096,
            "environment.seed": 99,
        }

    def test_round_trip_save_and_load(self, rl_config: RLRunConfig, tmp_path: Path) -> None:
        path = rl_config.save(tmp_path / "config.json")
        loaded = RLRunConfig.load(path)

        assert loaded == rl_config


class TestRLRunConfigGraphIntegration:
    """Tests for ExperimentGraph usage with nested RL configs."""

    def test_fork_applies_nested_algorithm_override(self, tmp_path: Path, rl_config: RLRunConfig) -> None:
        graph: ExperimentGraph[RLRunConfig] = ExperimentGraph(tmp_path / "exp", base_config=rl_config)
        root = graph.create_root(config=rl_config, branch="main", label="baseline")
        graph.fork(
            "lr_exp",
            from_node=root,
            config=rl_config.apply_overrides({"algorithm.learning_rate": 1e-5}),
        )

        resolved = graph.resolve_config(graph.head("lr_exp"))

        assert resolved.algorithm.learning_rate == 1e-5
        assert resolved.algorithm.total_timesteps == rl_config.algorithm.total_timesteps
        assert resolved.environment.nr_envs == rl_config.environment.nr_envs

    def test_extend_persists_resolved_nested_config(self, tmp_path: Path, rl_config: RLRunConfig) -> None:
        graph: ExperimentGraph[RLRunConfig] = ExperimentGraph(tmp_path / "exp", base_config=rl_config)
        graph.create_root(config=rl_config, branch="main", label="baseline")
        child = graph.extend(
            "main",
            config=rl_config.apply_overrides({"environment.nr_envs": 32}),
        )

        loaded = RLRunConfig.load(child.config_path)

        assert loaded.environment.nr_envs == 32
        assert loaded.algorithm.minibatch_size == rl_config.algorithm.minibatch_size
