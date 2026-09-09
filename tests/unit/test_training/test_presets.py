"""Unit tests for ``jarl.training.presets``."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from jarl.training.config import RLRunConfig
from jarl.training.launch import write_run_config
from jarl.training.presets import (
    PPO_ALGORITHM_NAME,
    PPO_GRU_ALGORITHM_NAME,
    RunFormPayload,
    child_config_overrides_from_payload,
    default_form_values,
    form_values_from_run_config,
    form_values_to_run_config,
    preset_run_config,
    recommended_video_frequency,
    run_overrides_from_payload,
    validate_run_form_values,
)


@pytest.mark.parametrize(
    ("algorithm", "expected_name"),
    [
        ("ppo", PPO_ALGORITHM_NAME),
        ("ppo_gru", PPO_GRU_ALGORITHM_NAME),
    ],
)
def test_preset_run_config_algorithm_name(algorithm: str, expected_name: str) -> None:
    config = preset_run_config(algorithm=algorithm)  # type: ignore[arg-type]
    assert config.algorithm.name == expected_name
    assert config.algorithm.total_timesteps == 512


def test_recommended_video_frequency_short_run() -> None:
    assert recommended_video_frequency(total_timesteps=512, eval_frequency=128) == 128


def test_default_form_values_are_production_scale() -> None:
    values = default_form_values()
    assert values.preset == "custom"
    assert values.total_timesteps == 5_000_000
    assert values.nr_envs == 64
    assert values.nr_steps == 128
    assert values.minibatch_size == 2048
    assert values.eval_frequency == 65_536
    assert values.nr_epochs == 4


def test_form_values_fast_preset_matches_preset_run_config() -> None:
    values = default_form_values(algorithm="ppo_gru", preset="fast")
    payload = RunFormPayload(values=values, video_frequency=128, record_final_video=True)
    config = form_values_to_run_config(payload)
    expected = preset_run_config(algorithm="ppo_gru").apply_overrides(
        {
            "tracking.track_wandb": False,
            "tracking.track_tensorboard": True,
            "tracking.wandb_project": "jarl",
            "tracking.wandb_mode": "offline",
            "video.record_video": False,
            "video.record_final_video": True,
            "video.video_frequency": 128,
            "video.video_episodes": 1,
        }
    )
    dumped = config.model_dump()
    expected_dump = expected.model_dump()
    dumped.pop("run_name")
    expected_dump.pop("run_name")
    assert dumped == expected_dump


def test_form_values_to_run_config_respects_max_episode_steps() -> None:
    values = replace(default_form_values(), max_episode_steps=100)
    payload = RunFormPayload(values=values, video_frequency=64, record_final_video=False)
    config = form_values_to_run_config(payload)
    assert config.environment.max_episode_steps == 100


def test_form_values_to_run_config_fast_preset_can_override_max_episode_steps() -> None:
    values = replace(default_form_values(preset="fast"), max_episode_steps=32)
    payload = RunFormPayload(values=values, video_frequency=128, record_final_video=True)
    config = form_values_to_run_config(payload)
    assert config.environment.max_episode_steps == 32


def test_form_values_to_run_config_respects_save_model_toggle() -> None:
    values = replace(default_form_values(), save_model=False)
    payload = RunFormPayload(values=values, video_frequency=64, record_final_video=False)
    config = form_values_to_run_config(payload)
    assert config.runner.save_model is False


def test_form_values_to_run_config_accepts_unset_video_frequency() -> None:
    values = replace(default_form_values(), record_video=False)
    payload = RunFormPayload(values=values, video_frequency=0, record_final_video=True)
    config = form_values_to_run_config(payload)
    assert config.video.video_frequency >= 1
    assert config.video.record_video is False


def test_form_values_to_run_config_fills_video_frequency_when_recording() -> None:
    values = replace(default_form_values(), record_video=True, preset="fast")
    payload = RunFormPayload(values=values, video_frequency=0, record_final_video=True)
    config = form_values_to_run_config(payload)
    assert config.video.record_video is True
    assert config.video.video_frequency >= 1


def test_run_overrides_respect_save_model_toggle() -> None:
    values = replace(default_form_values(), save_model=False)
    payload = RunFormPayload(values=values, video_frequency=64, record_final_video=False)
    overrides = run_overrides_from_payload(payload)
    assert overrides["runner.save_model"] is False


def test_run_overrides_exclude_environment() -> None:
    values = default_form_values()
    payload = RunFormPayload(values=values, video_frequency=64, record_final_video=False)
    overrides = run_overrides_from_payload(payload)
    assert "environment.env_id" not in overrides
    assert "algorithm.name" not in overrides
    assert overrides["algorithm.total_timesteps"] == values.total_timesteps


def test_write_run_config_roundtrip(tmp_path: Path) -> None:
    config = preset_run_config()
    path = tmp_path / "config.json"
    write_run_config(config, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["algorithm"]["total_timesteps"] == 512


def test_default_form_values_save_model_is_true() -> None:
    assert default_form_values().save_model is True


def test_form_values_from_run_config_preserves_algorithm_and_save_model() -> None:
    config = preset_run_config(algorithm="ppo_gru")
    values = form_values_from_run_config(config)
    assert values.algorithm == "ppo_gru"
    assert values.save_model is True
    assert values.nr_envs == config.environment.nr_envs


def test_child_config_overrides_include_env_id_when_map_changes() -> None:
    parent = preset_run_config().apply_overrides({"environment.env_id": "Navix-Empty-5x5-v0"})
    values = replace(form_values_from_run_config(parent), env_id="Navix-DoorKey-8x8-v0")
    payload = RunFormPayload(values=values, video_frequency=128, record_final_video=True)
    overrides = child_config_overrides_from_payload(parent, payload)
    assert overrides["environment.env_id"] == "Navix-DoorKey-8x8-v0"


def test_child_config_overrides_omit_env_id_when_unchanged() -> None:
    parent = preset_run_config().apply_overrides({"environment.env_id": "Navix-Empty-5x5-v0"})
    payload = RunFormPayload(
        values=form_values_from_run_config(parent),
        video_frequency=128,
        record_final_video=True,
    )
    overrides = child_config_overrides_from_payload(parent, payload)
    assert "environment.env_id" not in overrides


def test_child_config_overrides_include_max_episode_steps_when_changed() -> None:
    parent = preset_run_config()
    values = replace(form_values_from_run_config(parent), max_episode_steps=100)
    payload = RunFormPayload(values=values, video_frequency=128, record_final_video=True)
    overrides = child_config_overrides_from_payload(parent, payload)
    assert overrides["environment.max_episode_steps"] == 100


def test_child_config_overrides_omit_max_episode_steps_when_unchanged() -> None:
    parent = preset_run_config()
    payload = RunFormPayload(
        values=form_values_from_run_config(parent),
        video_frequency=128,
        record_final_video=True,
    )
    overrides = child_config_overrides_from_payload(parent, payload)
    assert "environment.max_episode_steps" not in overrides


def test_child_config_overrides_reject_incompatible_map() -> None:
    from jarl.envs.navix.catalog import NavixMapContract, register_map

    register_map(
        NavixMapContract(
            env_id="Navix-Incompatible-Test-v0",
            processed_obs_shape=(99,),
            action_names=("a", "b"),
            transfer_group="other_test_group",
        )
    )
    parent = preset_run_config().apply_overrides({"environment.env_id": "Navix-Empty-5x5-v0"})
    values = replace(form_values_from_run_config(parent), env_id="Navix-Incompatible-Test-v0")
    payload = RunFormPayload(values=values, video_frequency=128, record_final_video=True)
    with pytest.raises(ValueError, match="transfer"):
        child_config_overrides_from_payload(parent, payload)


def test_child_config_overrides_validate_parent_nr_envs() -> None:
    parent = preset_run_config().apply_overrides(
        {
            "environment.nr_envs": 8,
            "algorithm.nr_steps": 64,
            "algorithm.minibatch_size": 128,
        }
    )
    values = replace(form_values_from_run_config(parent), minibatch_size=1024)
    payload = RunFormPayload(values=values, video_frequency=128, record_final_video=True)
    with pytest.raises(ValueError, match="minibatch_size"):
        child_config_overrides_from_payload(parent, payload)


def test_validate_run_form_values_uses_parent_nr_envs() -> None:
    values = replace(
        default_form_values(),
        nr_steps=64,
        minibatch_size=1024,
    )
    assert validate_run_form_values(values) == []
    messages = validate_run_form_values(values, environment_nr_envs=8)
    assert len(messages) == 1
    assert "minibatch_size" in messages[0]


def test_run_overrides_validate_reference_nr_envs() -> None:
    values = replace(
        default_form_values(),
        nr_steps=128,
        minibatch_size=2048,
    )
    payload = RunFormPayload(
        values=values,
        video_frequency=64,
        record_final_video=False,
        reference_nr_envs=8,
    )
    with pytest.raises(ValueError, match="minibatch_size"):
        run_overrides_from_payload(payload)


def test_run_overrides_exclude_lineage_architecture() -> None:
    values = default_form_values()
    payload = RunFormPayload(values=values, video_frequency=64, record_final_video=False)
    overrides = run_overrides_from_payload(payload)
    assert not set(overrides) & RLRunConfig.immutable_paths()
