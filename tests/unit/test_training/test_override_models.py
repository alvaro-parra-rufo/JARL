"""Tests for typed sparse configuration override models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarl.training.config import RLRunConfig
from jarl.training.override_models import (
    ForkConfigOverrides,
    RetrainConfigOverrides,
    SparseConfigOverrides,
    build_sparse_config_overrides_model,
)


@pytest.mark.parametrize(
    ("model", "expected_paths"),
    [
        pytest.param(
            RetrainConfigOverrides,
            RLRunConfig.RETRAIN_MUTABLE_PATHS,
            id="retrain",
        ),
        pytest.param(
            ForkConfigOverrides,
            RLRunConfig.RETRAIN_MUTABLE_PATHS | RLRunConfig.FORK_EXTRA_MUTABLE_PATHS,
            id="fork",
        ),
    ],
)
def test_override_model_schema_matches_policy_paths(
    model: type[SparseConfigOverrides],
    expected_paths: frozenset[str],
) -> None:
    schema = model.model_json_schema()

    assert frozenset(schema["properties"]) == expected_paths
    assert "required" not in schema
    assert schema["additionalProperties"] is False


def test_override_model_schema_reuses_config_type_constraints_and_description() -> None:
    properties = ForkConfigOverrides.model_json_schema()["properties"]

    assert properties["algorithm.total_timesteps"]["type"] == "integer"
    assert properties["algorithm.total_timesteps"]["minimum"] == 1
    assert properties["algorithm.total_timesteps"]["description"]
    assert properties["algorithm.learning_rate"]["type"] == "number"
    assert properties["algorithm.learning_rate"]["exclusiveMinimum"] == 0.0


def test_override_model_serializes_only_explicit_dotted_values() -> None:
    overrides = ForkConfigOverrides.model_validate(
        {
            "algorithm.total_timesteps": 1024,
            "environment.env_id": "Navix-DoorKey-5x5-v0",
        }
    )

    payload = overrides.to_dotted_dict()

    assert payload == {
        "algorithm.total_timesteps": 1024,
        "environment.env_id": "Navix-DoorKey-5x5-v0",
    }


def test_override_model_preserves_explicit_none() -> None:
    overrides = RetrainConfigOverrides.model_validate({"tracking.wandb_mode": None})

    payload = overrides.to_dotted_dict()

    assert payload == {"tracking.wandb_mode": None}


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"algorithm.total_timesteps": 0}, id="constraint"),
        pytest.param({"algorithm.total_timesteps": "many"}, id="type"),
        pytest.param({"algorithm.name": "ppo.full_jax.navix"}, id="unknown-path"),
    ],
)
def test_override_model_rejects_invalid_values(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RetrainConfigOverrides.model_validate(payload)


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("environment.reward.goal_reached", id="reward-weight"),
        pytest.param("environment.scenario_reward_id", id="scenario-id"),
    ],
)
def test_override_models_do_not_expose_reward_or_overlay_paths(path: str) -> None:
    schema = ForkConfigOverrides.model_json_schema()

    assert path not in schema["properties"]
    with pytest.raises(ValidationError):
        ForkConfigOverrides.model_validate({path: 1.0})


def test_override_model_builder_rejects_unknown_config_path() -> None:
    with pytest.raises(ValueError, match="Unknown config override path"):
        build_sparse_config_overrides_model(
            "BrokenOverrides",
            config_cls=RLRunConfig,
            allowed_paths=frozenset({"algorithm.missing"}),
        )
