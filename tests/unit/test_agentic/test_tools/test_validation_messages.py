"""Tests for tool request validation messages."""

from __future__ import annotations

from pydantic import ValidationError

from jarl.agentic.tools._validation_messages import (
    format_config_overrides_catalog,
    format_request_validation_error,
)
from jarl.agentic.tools.graph.extend import ToolRequest
from jarl.agentic.tools.train.run import ToolRequest as TrainRunToolRequest
from jarl.training.override_models import ForkConfigOverrides, RetrainConfigOverrides


def test_format_config_overrides_catalog_lists_keys_and_descriptions() -> None:
    catalog = format_config_overrides_catalog(ForkConfigOverrides)

    assert "Allowed config_overrides keys" in catalog
    assert "algorithm.entropy_coef: Entropy bonus coefficient." in catalog
    assert "algorithm.evaluation_and_save_frequency:" in catalog


def test_format_request_validation_error_lists_schema_catalog_without_aliases() -> None:
    try:
        ToolRequest.model_validate(
            {
                "config_overrides": {
                    "entropy_coefficient": 0.08,
                    "learning_rate": 0.001,
                },
            }
        )
    except ValidationError as exc:
        message = format_request_validation_error(ToolRequest, exc)
    else:
        raise AssertionError("Expected validation to fail.")

    assert "Invalid config_overrides keys:" in message
    assert "entropy_coefficient" in message
    assert "learning_rate" in message
    assert message.count("Extra inputs are not permitted") == 0
    assert "→" not in message
    assert "algorithm.entropy_coef: Entropy bonus coefficient." in message
    assert "algorithm.learning_rate: Initial PPO learning rate." in message


def test_format_request_validation_error_eval_frequency_shows_catalog_not_alias() -> None:
    try:
        ToolRequest.model_validate(
            {
                "config_overrides": {"algorithm.eval_frequency": 128},
            }
        )
    except ValidationError as exc:
        message = format_request_validation_error(ToolRequest, exc)
    else:
        raise AssertionError("Expected validation to fail.")

    assert "algorithm.eval_frequency" in message
    assert "→" not in message
    assert "algorithm.evaluation_and_save_frequency:" in message
    assert "evaluation and checkpoint" in message.lower()


def test_format_request_validation_error_groups_nested_override_keys() -> None:
    try:
        ToolRequest.model_validate(
            {
                "config_overrides": {
                    "ppo.clip_range": 0.15,
                    "ppo.lambda": 0.9,
                },
            }
        )
    except ValidationError as exc:
        message = format_request_validation_error(ToolRequest, exc)
    else:
        raise AssertionError("Expected validation to fail.")

    assert "Invalid config_overrides keys: ppo.clip_range, ppo.lambda." in message
    assert message.count("Extra inputs are not permitted") == 0
    assert "Allowed config_overrides keys" in message


def test_format_request_validation_error_uses_retrain_catalog_for_train_run() -> None:
    try:
        TrainRunToolRequest.model_validate(
            {
                "config_overrides": {"environment.env_id": "Navix-Empty-5x5-v0"},
            }
        )
    except ValidationError as exc:
        message = format_request_validation_error(TrainRunToolRequest, exc)
    else:
        raise AssertionError("Expected validation to fail.")

    retrain_catalog = format_config_overrides_catalog(RetrainConfigOverrides)
    fork_catalog = format_config_overrides_catalog(ForkConfigOverrides)
    assert retrain_catalog in message
    assert fork_catalog not in message
