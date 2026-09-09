"""Tests for session agentic tools."""

from __future__ import annotations

import json
from pathlib import Path

from jarl.agentic.audit import audit_index_path
from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.registry import REGISTRY
from jarl.agentic.tools.session.finish import ToolRequest as FinishToolRequest
from jarl.agentic.tools.session.finish import run_session_finish
from jarl.agentic.tools.session.status import ToolRequest, run_session_status
from jarl.agentic.workflow import AgenticWorkflow
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeStatus
from jarl.training.config import RLRunConfig


class TestSessionStatusTool:
    def test_status_includes_active_node(self, tool_context: ToolContext) -> None:
        payload = json.loads(run_session_status(tool_context, ToolRequest()))

        assert payload["thread_id"] == tool_context.thread_id
        assert payload["active_node_id"] == tool_context.current_node_id
        assert payload["active_status"] == "prepared"
        assert payload["node_count"] == 1
        assert payload["current_branch"] == "main"
        assert payload["branch_heads"]["main"] == tool_context.current_node_id
        assert payload["current_branch_summary"]["branch"] == "main"
        assert len(payload["current_branch_summary"]["nodes"]) == 1

    def test_status_without_active_node(self, empty_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(empty_experiment)
        ctx = workflow.build_tool_context()

        payload = json.loads(run_session_status(ctx, ToolRequest()))

        assert payload["active_node_id"] is None
        assert payload["active_status"] is None
        assert payload["node_count"] == 0
        assert payload["current_branch"] is None
        assert payload["current_branch_summary"] is None


class TestSessionAudit:
    def test_session_status_audits_when_reads_enabled(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment, audit_reads=True)
        handler = REGISTRY.get("session_status").build_handler(workflow)

        handler({})

        event = json.loads(audit_index_path(prepared_experiment).read_text(encoding="utf-8").splitlines()[0])
        assert event["tool"] == "session_status"
        assert event["kind"] == "read"

    def test_status_handler_reloads_disk_status(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        handler = REGISTRY.get("session_status").build_handler(workflow)
        first = json.loads(handler({}))
        graph = ExperimentGraph.from_directory(prepared_experiment, config_cls=RLRunConfig)
        node = graph.get_node(first["active_node_id"])
        node._meta.status = NodeStatus.TRAINING
        node._save_metadata()

        second = json.loads(handler({}))

        assert first["active_status"] == NodeStatus.PREPARED.value
        assert second["active_status"] == NodeStatus.TRAINING.value


class TestSessionFinishTool:
    def test_finish_declares_finished_payload(self, tool_context: ToolContext) -> None:
        payload = json.loads(run_session_finish(tool_context, FinishToolRequest()))

        assert payload == {
            "finished": True,
            "thread_id": tool_context.thread_id,
        }

    def test_session_finish_audits_when_reads_enabled(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment, audit_reads=True)
        handler = REGISTRY.get("session_finish").build_handler(workflow)

        handler({})

        event = json.loads(audit_index_path(prepared_experiment).read_text(encoding="utf-8").splitlines()[0])
        assert event["tool"] == "session_finish"
        assert event["kind"] == "read"
        assert event["result"]["finished"] is True
