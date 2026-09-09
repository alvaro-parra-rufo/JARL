"""Deterministic handler behavior for ``graph_fork``."""

from __future__ import annotations

import json

import pytest

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.registry import REGISTRY


class TestGraphForkBehavior:
    def test_handler_rejects_config_path_outside_typed_schema(self, navix_tool_context: ToolContext) -> None:
        root_id = navix_tool_context.graph.current_node.id
        handler = REGISTRY.get("graph_fork").build_handler(navix_tool_context.workflow)

        result = json.loads(
            handler(
                {
                    "branch": "exp",
                    "label": "bad",
                    "prepare": True,
                    "from_node": root_id,
                    "config_overrides": {"algorithm.name": "ppo.full_jax.navix"},
                }
            )
        )

        assert result["code"] == "validation"
        assert "algorithm.name" in result["error"]

    @pytest.mark.parametrize(
        "invalid_overrides",
        [
            pytest.param({"algorithm.eval_frequency": 128}, id="highlight-shorthand-eval"),
            pytest.param({"algorithm.nr_envs": 16}, id="wrong-path-nr-envs"),
        ],
    )
    def test_handler_rejects_invalid_override_keys(
        self,
        navix_tool_context: ToolContext,
        invalid_overrides: dict[str, object],
    ) -> None:
        root_id = navix_tool_context.graph.current_node.id
        handler = REGISTRY.get("graph_fork").build_handler(navix_tool_context.workflow)

        result = json.loads(
            handler(
                {
                    "branch": "exp",
                    "label": "bad",
                    "prepare": True,
                    "from_node": root_id,
                    "config_overrides": invalid_overrides,
                }
            )
        )

        assert result["code"] == "validation"
        assert next(iter(invalid_overrides)) in result["error"]
