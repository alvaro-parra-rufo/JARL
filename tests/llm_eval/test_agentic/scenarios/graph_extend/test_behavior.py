"""Deterministic behavior scenarios for ``graph_extend`` (manual EX-1…EX-3)."""

from __future__ import annotations

import json

import pytest

from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.graph.extend import ToolRequest, run_graph_extend
from jarl.agentic.tools.registry import REGISTRY
from jarl.operations.graph.checkout import CheckoutRequest, checkout
from jarl.operations.graph.fork import ForkRequest, fork


class TestGraphExtendBehavior:
    def test_ex1_extend_anchors_to_branch_head(self, navix_tool_context: ToolContext) -> None:
        graph = navix_tool_context.graph
        root_id = graph.current_node.id
        first = fork(graph, ForkRequest(branch="exp", label="first", prepare=True))
        second_payload = json.loads(
            run_graph_extend(navix_tool_context, ToolRequest(branch="exp", label="second", prepare=True))
        )
        checkout(graph, CheckoutRequest(node_id=root_id))

        third_payload = json.loads(
            run_graph_extend(navix_tool_context, ToolRequest(branch="exp", label="third", prepare=True))
        )

        assert third_payload["parent_id"] == second_payload["node_id"]
        assert first.node_id != third_payload["parent_id"]

    def test_ex2_extend_keeps_same_branch_with_timesteps(self, navix_tool_context: ToolContext) -> None:
        graph = navix_tool_context.graph

        payload = json.loads(
            run_graph_extend(
                navix_tool_context,
                ToolRequest(
                    branch="main",
                    label="continued",
                    prepare=True,
                    config_overrides={"algorithm.total_timesteps": 1024},
                ),
            )
        )

        assert payload["branch"] == "main"
        child = graph.get_node(payload["node_id"])

        assert graph.resolve_config(child).algorithm.total_timesteps == 1024

    def test_ex3_extend_allows_environment_env_id_on_same_branch(self, navix_tool_context: ToolContext) -> None:
        graph = navix_tool_context.graph

        payload = json.loads(
            run_graph_extend(
                navix_tool_context,
                ToolRequest(
                    branch="main",
                    label="doorkey-line",
                    prepare=True,
                    config_overrides={"environment.env_id": "Navix-DoorKey-5x5-v0"},
                ),
            )
        )

        child = graph.get_node(payload["node_id"])

        assert graph.resolve_config(child).environment.env_id == "Navix-DoorKey-5x5-v0"

    def test_handler_rejects_config_path_outside_typed_schema(self, navix_tool_context: ToolContext) -> None:
        handler = REGISTRY.get("graph_extend").build_handler(navix_tool_context.workflow)

        result = json.loads(
            handler(
                {
                    "branch": "main",
                    "label": "bad",
                    "prepare": True,
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
        handler = REGISTRY.get("graph_extend").build_handler(navix_tool_context.workflow)

        result = json.loads(
            handler(
                {
                    "branch": "main",
                    "label": "bad",
                    "prepare": True,
                    "config_overrides": invalid_overrides,
                }
            )
        )

        assert result["code"] == "validation"
        assert next(iter(invalid_overrides)) in result["error"]
