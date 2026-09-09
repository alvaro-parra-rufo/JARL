"""Deterministic handler behavior for ``train_run``."""

from __future__ import annotations

import json

import pytest

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.registry import REGISTRY


class TestTrainRunBehavior:
    def test_handler_rejects_config_path_outside_typed_schema(self, navix_tool_context: ToolContext) -> None:
        handler = REGISTRY.get("train_run").build_handler(navix_tool_context.workflow)

        result = json.loads(
            handler(
                {
                    "config_overrides": {"environment.env_id": "Navix-DoorKey-5x5-v0"},
                }
            )
        )

        assert result["code"] == "validation"
        assert "environment.env_id" in result["error"]

    @pytest.mark.parametrize(
        "invalid_overrides",
        [
            pytest.param({"algorithm.eval_frequency": 128}, id="highlight-shorthand-eval"),
            pytest.param({"algorithm.name": "ppo.full_jax.navix"}, id="immutable-algorithm-name"),
        ],
    )
    def test_handler_rejects_invalid_override_keys(
        self,
        navix_tool_context: ToolContext,
        invalid_overrides: dict[str, object],
    ) -> None:
        handler = REGISTRY.get("train_run").build_handler(navix_tool_context.workflow)

        result = json.loads(
            handler(
                {
                    "config_overrides": invalid_overrides,
                }
            )
        )

        assert result["code"] == "validation"
        assert next(iter(invalid_overrides)) in result["error"]
