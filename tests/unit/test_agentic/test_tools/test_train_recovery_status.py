"""Tests for train recovery_status agentic tool."""

from __future__ import annotations

import json
from pathlib import Path

from jarl.agentic.audit import audit_index_path
from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.registry import REGISTRY
from jarl.agentic.tools.train.recovery_status import ToolRequest, run_train_recovery_status
from jarl.agentic.workflow import AgenticWorkflow
from jarl.experiments.node import NodeStatus


def _fail_with_checkpoint(tool_context: ToolContext, step: int = 4) -> None:
    workspace = tool_context.graph.current_node
    workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
    try:
        with workspace:
            workspace.save_checkpoint(
                step,
                {"value": step},
                metrics={"eval/episode_return": 0.2},
                force=True,
            )
            raise RuntimeError("tool boom")
    except RuntimeError:
        pass
    tool_context.graph.save()


class TestTrainRecoveryStatusTool:
    def test_run_reports_train_resume(self, tool_context: ToolContext) -> None:
        _fail_with_checkpoint(tool_context, step=4)

        payload = json.loads(run_train_recovery_status(tool_context, ToolRequest()))

        assert payload["node_status"] == NodeStatus.FAILED.value
        assert payload["recommended_action"] == "train_resume"
        assert payload["can_resume"] is True
        assert payload["resumable_checkpoint_step"] == 4
        assert "resume_hint" not in payload


class TestTrainRecoveryStatusAudit:
    def test_audits_as_read(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment, audit_reads=True)
        workspace = workflow.graph.current_node
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        try:
            with workspace:
                workspace.save_checkpoint(2, {"value": 2}, force=True)
                raise RuntimeError("audit boom")
        except RuntimeError:
            pass
        workflow.graph.save()
        handler = REGISTRY.get("train_recovery_status").build_handler(workflow)

        handler({})

        event = json.loads(audit_index_path(prepared_experiment).read_text(encoding="utf-8").splitlines()[0])
        assert event["tool"] == "train_recovery_status"
        assert event["kind"] == "read"
        assert event["result"]["recommended_action"] == "train_resume"
