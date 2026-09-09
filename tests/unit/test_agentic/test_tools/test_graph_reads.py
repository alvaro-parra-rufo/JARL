"""Tests for graph read agentic tools."""

from __future__ import annotations

import json
from pathlib import Path

from jarl.agentic.audit import audit_index_path
from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.graph.checkpoints import ToolRequest as CheckpointsToolRequest
from jarl.agentic.tools.graph.checkpoints import run_graph_checkpoints
from jarl.agentic.tools.graph.diff import ToolRequest as DiffToolRequest
from jarl.agentic.tools.graph.diff import run_graph_diff
from jarl.agentic.tools.graph.reward import ToolRequest as RewardToolRequest
from jarl.agentic.tools.graph.reward import run_graph_reward
from jarl.agentic.tools.graph.set_reward import ToolRequest as SetRewardToolRequest
from jarl.agentic.tools.graph.set_reward import run_graph_set_reward
from jarl.agentic.tools.graph.subtree import ToolRequest as SubtreeToolRequest
from jarl.agentic.tools.graph.subtree import run_graph_subtree
from jarl.agentic.tools.graph.summary import ToolRequest as SummaryToolRequest
from jarl.agentic.tools.graph.summary import run_graph_summary
from jarl.agentic.tools.registry import REGISTRY
from jarl.agentic.workflow import AgenticWorkflow
from jarl.experiments.io.checkpoints import CHECKPOINT_BEST_RETURN_METRIC
from jarl.training.config import RLRunConfig


class TestGraphSummaryTool:
    def test_summary_includes_config_highlights(self, tool_context: ToolContext) -> None:
        payload = json.loads(run_graph_summary(tool_context, SummaryToolRequest()))

        assert payload["experiment"]["node_count"] == 1
        assert payload["nodes"][0]["config_highlights"]["env"] == RLRunConfig().environment.env_id
        assert "checkpoints" not in payload["nodes"][0]

    def test_summary_filters_single_node(self, tool_context: ToolContext) -> None:
        node_id = tool_context.current_node_id

        payload = json.loads(run_graph_summary(tool_context, SummaryToolRequest(node_id=node_id)))

        assert len(payload["nodes"]) == 1
        assert payload["nodes"][0]["node_id"] == node_id


class TestGraphRewardTool:
    def test_native_mix_is_null(self, tool_context: ToolContext) -> None:
        payload = json.loads(run_graph_reward(tool_context, RewardToolRequest()))

        assert payload == {"node_id": tool_context.current_node_id, "reward": None}

    def test_custom_mix_returns_catalog_only(self, tool_context: ToolContext) -> None:
        json.loads(run_graph_set_reward(tool_context, SetRewardToolRequest(goal_reached=4.0)))

        payload = json.loads(run_graph_reward(tool_context, RewardToolRequest()))
        dumped = str(payload).lower()

        assert payload["reward"]["goal_reached"] == 4.0
        assert "cell_entry" not in payload["reward"]
        assert "scenario_reward" not in dumped
        assert "floor_cell" not in dumped
        assert "dopamina" not in dumped

    def test_handler_rejects_cell_entry(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment, audit_reads=False)
        handler = REGISTRY.get("graph_reward").build_handler(workflow)

        result = json.loads(handler({"cell_entry": 0.5}))

        assert result["code"] == "validation"
        assert "cell_entry" in result["error"]


class TestGraphCheckpointsTool:
    def test_overview_for_current_node(self, tool_context: ToolContext) -> None:
        workspace = tool_context.graph.current_node
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        workspace.save_checkpoint(
            10,
            {"value": 10},
            metrics={CHECKPOINT_BEST_RETURN_METRIC: 0.7},
            force=True,
        )
        tool_context.graph.save()

        payload = json.loads(run_graph_checkpoints(tool_context, CheckpointsToolRequest()))

        assert payload["mode"] == "overview"
        assert payload["checkpoint_count"] == 1
        assert payload["latest"]["checkpoint_step"] == 10
        assert payload["latest"]["eval_return"] == 0.7

    def test_search_returns_candidates(self, tool_context: ToolContext) -> None:
        workspace = tool_context.graph.current_node
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        workspace.save_checkpoint(5, {"value": 5}, metrics={CHECKPOINT_BEST_RETURN_METRIC: 0.2}, force=True)
        workspace.save_checkpoint(15, {"value": 15}, metrics={CHECKPOINT_BEST_RETURN_METRIC: 0.8}, force=True)
        tool_context.graph.save()

        payload = json.loads(
            run_graph_checkpoints(
                tool_context,
                CheckpointsToolRequest(sort_by=CHECKPOINT_BEST_RETURN_METRIC, limit=1),
            )
        )

        assert payload["mode"] == "search"
        assert payload["returned_count"] == 1
        assert payload["candidates"][0]["checkpoint_step"] == 15
        assert payload["has_more"] is True


class TestGraphSubtreeTool:
    def test_subtree_returns_root_node(self, tool_context: ToolContext) -> None:
        payload = json.loads(run_graph_subtree(tool_context, SubtreeToolRequest(depth=1)))

        assert payload["root_id"] == tool_context.current_node_id
        assert len(payload["nodes"]) == 1


class TestGraphDiffTool:
    def test_diff_reports_override(self, tool_context: ToolContext) -> None:
        graph = tool_context.graph
        parent_id = graph.current_node.id
        child = graph.fork(
            "exp",
            from_node=graph.current_node,
            config=RLRunConfig().apply_overrides({"algorithm.learning_rate": 1e-5}),
            prepare=True,
        )
        graph.save()

        payload = json.loads(
            run_graph_diff(
                tool_context,
                DiffToolRequest(node_a=parent_id, node_b=child.id),
            )
        )

        assert "algorithm" in payload["diff"]["changed"]


class TestGraphRewardAudit:
    def test_graph_reward_handler_audits_when_reads_enabled(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment, audit_reads=True)
        handler = REGISTRY.get("graph_reward").build_handler(workflow)

        handler({})

        event = json.loads(audit_index_path(prepared_experiment).read_text(encoding="utf-8").splitlines()[0])
        assert event["tool"] == "graph_reward"
        assert event["kind"] == "read"
        assert event["result"]["reward"] is None
