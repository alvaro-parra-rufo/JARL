"""Tests for ``jarl.agentic.errors``."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from jarl.agentic.errors import AGENTIC_ENVIRONMENT_CONFIG_ERROR, AgenticError, ToolError
from jarl.envs.navix.scenario_rewards import ScenarioRewardError


class _SampleRequest(BaseModel):
    required: str


class TestToolError:
    def test_from_exception_wraps_value_error(self) -> None:
        error = ToolError.from_exception("graph_fork", ValueError("bad override"))

        assert error.tool == "graph_fork"
        assert error.code == "domain"
        assert error.message == "bad override"
        assert isinstance(error.cause, ValueError)

    def test_from_exception_preserves_existing_tool_error(self) -> None:
        original = ToolError(tool="train_run", code="internal", message="boom")

        assert ToolError.from_exception("graph_fork", original) is original

    def test_from_exception_maps_scenario_reward_error_to_neutral_message(self) -> None:
        cause = ScenarioRewardError(
            "ScenarioRewardSpec navix.floor_cell v1 incompatible with env_id 'Navix-Empty-5x5-v0'."
        )

        error = ToolError.from_exception("train_run", cause)
        payload = {"error": error.message, "code": error.code}

        assert error.code == "domain"
        assert error.message == AGENTIC_ENVIRONMENT_CONFIG_ERROR
        assert "navix.floor_cell" not in error.message
        assert "compatible_env_ids" not in error.message
        assert "Navix-EmptyVariant-5x5-v0" not in error.message
        assert payload["error"] == AGENTIC_ENVIRONMENT_CONFIG_ERROR
        assert error.cause is cause

    @pytest.mark.parametrize(
        ("exc", "expected_code"),
        [
            (ValidationError.from_exception_data("x", []), "validation"),
            (AgenticError("agentic"), "domain"),
            (KeyError("node"), "domain"),
            (RuntimeError("oops"), "domain"),
            (TypeError("unexpected"), "internal"),
        ],
    )
    def test_code_for_exception(self, exc: Exception, expected_code: str) -> None:
        assert ToolError.code_for_exception(exc) == expected_code
