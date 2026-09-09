"""Tests for graph mutation agentic tools."""

from __future__ import annotations

import json
from pathlib import Path

from jarl.agentic.audit import audit_index_path
from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.graph.checkout import ToolRequest as CheckoutToolRequest
from jarl.agentic.tools.graph.checkout import run_graph_checkout
from jarl.agentic.tools.graph.extend import ToolRequest as ExtendToolRequest
from jarl.agentic.tools.graph.extend import run_graph_extend
from jarl.agentic.tools.graph.fork import ToolRequest as ForkToolRequest
from jarl.agentic.tools.graph.fork import run_graph_fork
from jarl.agentic.tools.graph.set_reward import ToolRequest as SetRewardToolRequest
from jarl.agentic.tools.graph.set_reward import run_graph_set_reward
from jarl.agentic.tools.registry import REGISTRY
from jarl.agentic.workflow import AgenticWorkflow
from jarl.operations.graph.checkout import CheckoutRequest, checkout
from jarl.operations.graph.fork import ForkRequest, fork


class TestGraphCheckoutTool:
    def test_checkout_moves_current_node(self, tool_context: ToolContext) -> None:
        graph = tool_context.graph
        child = graph.fork("exp", from_node=graph.current_node, prepare=True)
        graph.save()

        payload = json.loads(run_graph_checkout(tool_context, CheckoutToolRequest(node_id=child.id)))

        assert payload["node_id"] == child.id
        assert tool_context.current_node_id == child.id


class TestGraphForkTool:
    def test_fork_creates_child(self, tool_context: ToolContext) -> None:
        parent_id = tool_context.current_node_id

        payload = json.loads(
            run_graph_fork(
                tool_context,
                ForkToolRequest(branch="exp", label="child", prepare=True),
            )
        )

        assert payload["parent_id"] == parent_id
        assert payload["branch"] == "exp"
        assert payload["status"] == "prepared"

    def test_fork_handler_rejects_config_path_outside_typed_schema(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment, audit_reads=False)
        root_id = workflow.current_node_id
        handler = REGISTRY.get("graph_fork").build_handler(workflow)

        result = json.loads(
            handler(
                {
                    "branch": "exp",
                    "label": "child",
                    "prepare": True,
                    "from_node": root_id,
                    "config_overrides": {"algorithm.name": "ppo.full_jax.navix"},
                }
            )
        )

        assert result["code"] == "validation"
        assert "algorithm.name" in result["error"]


class TestGraphExtendTool:
    def test_extend_anchors_to_branch_head_not_current_node(self, tool_context: ToolContext) -> None:
        """Extend must parent from the branch head even when current_node is an older node."""
        graph = tool_context.graph
        root_id = graph.current_node.id
        first = fork(graph, ForkRequest(branch="exp", label="first", prepare=True))
        second_payload = json.loads(
            run_graph_extend(
                tool_context,
                ExtendToolRequest(branch="exp", label="second", prepare=True),
            )
        )
        checkout(graph, CheckoutRequest(node_id=root_id))

        third_payload = json.loads(
            run_graph_extend(
                tool_context,
                ExtendToolRequest(branch="exp", label="third", prepare=True),
            )
        )

        assert third_payload["parent_id"] == second_payload["node_id"]
        assert first.node_id != third_payload["parent_id"]


class TestGraphSetRewardTool:
    def test_patches_current_node_in_place(self, tool_context: ToolContext) -> None:
        node_id = tool_context.current_node_id

        payload = json.loads(run_graph_set_reward(tool_context, SetRewardToolRequest(goal_reached=4.0)))

        assert payload["node_id"] == node_id
        assert payload["reward"]["goal_reached"] == 4.0
        assert payload["reward"]["key_pickup"] == 0.0
        assert tool_context.current_node_id == node_id
        assert tool_context.graph.as_networkx().number_of_nodes() == 1

    def test_handler_rejects_cell_entry(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment, audit_reads=False)
        handler = REGISTRY.get("graph_set_reward").build_handler(workflow)

        result = json.loads(handler({"cell_entry": 0.5}))

        assert result["code"] == "validation"
        assert "cell_entry" in result["error"]
        assert "floor_cell" not in result["error"]
        assert "compatible_env_ids" not in result["error"]

    def test_handler_rejects_negative_weight(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment, audit_reads=False)
        handler = REGISTRY.get("graph_set_reward").build_handler(workflow)

        result = json.loads(handler({"goal_reached": -1}))

        assert result["code"] == "validation"
        assert "26" not in result["error"]
        assert "floor_cell" not in result["error"]


class TestGraphMutationAudit:
    def test_graph_fork_handler_audits_mutation(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment, audit_reads=False)
        root_id = workflow.current_node_id
        handler = REGISTRY.get("graph_fork").build_handler(workflow)

        handler({"branch": "exp", "label": "child", "prepare": True, "from_node": root_id})

        event = json.loads(audit_index_path(prepared_experiment).read_text(encoding="utf-8").splitlines()[0])
        assert event["tool"] == "graph_fork"
        assert event["kind"] == "mutation"
        assert event["node_before"] == root_id
        assert event["node_after"] == workflow.current_node_id

    def test_graph_extend_handler_audits_mutation(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment, audit_reads=False)
        root_id = workflow.current_node_id
        handler = REGISTRY.get("graph_extend").build_handler(workflow)

        handler({"label": "continued", "prepare": True})

        event = json.loads(audit_index_path(prepared_experiment).read_text(encoding="utf-8").splitlines()[0])
        assert event["tool"] == "graph_extend"
        assert event["kind"] == "mutation"
        assert event["node_before"] == root_id
        assert event["node_after"] == workflow.current_node_id

    def test_graph_set_reward_handler_audits_mutation(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment, audit_reads=False)
        root_id = workflow.current_node_id
        handler = REGISTRY.get("graph_set_reward").build_handler(workflow)

        handler({"goal_reached": 4.0})

        event = json.loads(audit_index_path(prepared_experiment).read_text(encoding="utf-8").splitlines()[0])
        assert event["tool"] == "graph_set_reward"
        assert event["kind"] == "mutation"
        assert event["node_before"] == root_id
        assert event["node_after"] == root_id
