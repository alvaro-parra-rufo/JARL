"""Tests for ``RewardWeightsConfig``."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.envs.navix.rewards import CHANNEL_SPECS


class TestRewardWeightsConfig:
    """Schema constraints for the configurable Navix mix."""

    def test_defaults_are_task_only(self) -> None:
        weights = RewardWeightsConfig()

        assert weights.goal_reached == 1.0
        assert all(value == 0.0 for name, value in weights.model_dump().items() if name != "goal_reached")

    def test_schema_names_match_channel_specs(self) -> None:
        assert tuple(RewardWeightsConfig.model_fields) == tuple(spec.name for spec in CHANNEL_SPECS)
        assert len(CHANNEL_SPECS) == 21

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(0.0, id="zero"),
            pytest.param(0.5, id="half"),
            pytest.param(1.0, id="one"),
            pytest.param(1.1, id="slightly-above-one"),
            pytest.param(4.0, id="four"),
        ],
    )
    def test_accepts_non_negative_finite_weights(self, value: float) -> None:
        weights = RewardWeightsConfig(goal_reached=value, distance_to_goal=value)

        assert weights.goal_reached == value
        assert weights.distance_to_goal == value

    def test_accepts_several_weights_above_one(self) -> None:
        weights = RewardWeightsConfig(goal_reached=4.0, goal_approach=2.0, holding_key=1.1)

        assert weights.goal_reached == 4.0
        assert weights.goal_approach == 2.0
        assert weights.holding_key == 1.1

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(-0.1, id="negative"),
            pytest.param(math.nan, id="nan"),
            pytest.param(math.inf, id="inf"),
            pytest.param(-math.inf, id="neg-inf"),
        ],
    )
    def test_rejects_invalid_weights(self, value: float) -> None:
        with pytest.raises(ValidationError):
            RewardWeightsConfig(goal_reached=value)

    def test_rejects_unknown_and_cell_entry_fields(self) -> None:
        with pytest.raises(ValidationError):
            RewardWeightsConfig.model_validate({"goal_reached": 1.0, "cell_entry": 0.5})
        with pytest.raises(ValidationError):
            RewardWeightsConfig.model_validate({"invented": 1.0})

    def test_field_descriptions_are_geometric(self) -> None:
        leaked = ("26", "25.6", "16.6", "15.93", "dopamina", "bonus", "trap", "cell_entry")
        for name, field in RewardWeightsConfig.model_fields.items():
            description = field.description or ""
            assert description
            lowered = description.lower()
            for token in leaked:
                assert token not in lowered
            assert "cell_entry" not in name
