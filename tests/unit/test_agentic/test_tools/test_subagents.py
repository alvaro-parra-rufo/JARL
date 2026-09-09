"""Tests for subagent metrics-analysis tool and registry."""

from __future__ import annotations

import json
from pathlib import Path

from jarl.agentic.audit import audit_index_path, run_dir, subagent_run_dir
from jarl.agentic.tools.registry import REGISTRY
from jarl.agentic.tools.subagent.metrics_analysis import (
    ToolRequest as MetricsToolRequest,
)
from jarl.agentic.tools.subagent.metrics_analysis import (
    run_subagent_metrics_analysis,
)
from jarl.agentic.workflow import AgenticWorkflow
from jarl.operations.subagent.metrics_analysis import MetricsAnalysisRequest, metrics_analysis
from tests.helpers.metrics_analysis_fake import MetricsAnalysisFake


class TestSubagentMetricsAnalysisTool:
    def test_run_reports_labels_and_evidence(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        ctx = workflow.build_tool_context()
        workspace = ctx.graph.current_node
        for step in range(9):
            workspace.log_scalar(step, "eval/episode_return", float(step))
        ctx.graph.save()

        features = metrics_analysis(
            ctx.graph,
            MetricsAnalysisRequest(node_id=workspace.id),
        )
        fake = MetricsAnalysisFake(
            messages=iter(()),
            evidence_ids=[
                "eval_return.window_initial.mean",
                "eval_return.window_final.mean",
            ],
        )
        workflow.set_llm(fake)

        payload = json.loads(run_subagent_metrics_analysis(ctx, MetricsToolRequest()))

        assert "improving" in payload["labels"]
        assert payload["summary"]
        assert set(payload["evidence"]) <= set(features.feature_ids)

    def test_registers_nested_subagent_run(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        ctx = workflow.build_tool_context()
        workspace = ctx.graph.current_node
        workspace.log_scalar(0, "eval/episode_return", 1.0)
        workspace.log_scalar(1, "eval/episode_return", 2.0)
        workspace.log_scalar(2, "eval/episode_return", 3.0)
        ctx.graph.save()
        workflow.set_llm(
            MetricsAnalysisFake(
                messages=iter(()),
                evidence_ids=["eval_return.first.value", "eval_return.last.value"],
            )
        )

        run_subagent_metrics_analysis(ctx, MetricsToolRequest())

        child_thread = f"{workflow.thread_id}:metrics_analysis"
        child_root = subagent_run_dir(prepared_experiment, workflow.thread_id, "metrics_analysis")
        manifest = json.loads((child_root / "manifest.json").read_text())
        assert manifest["run_kind"] == "subagent"
        assert manifest["parent_tool"] == "subagent_metrics_analysis"
        assert manifest["thread_id"] == child_thread
        assert child_root.is_relative_to(run_dir(prepared_experiment, workflow.thread_id))


class TestSubagentRegistry:
    def test_registry_discovers_metrics_analysis(self) -> None:
        names = REGISTRY.names()

        assert "subagent_metrics_analysis" in names
        assert "subagent_summarize_metrics" not in names
        assert "subagent_summarize_checkpoint_events" not in names
        assert "train_recovery_status" in names

    def test_subagent_tool_audits_via_wire_handler(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment, audit_reads=False)
        workspace = workflow.graph.current_node
        for step in range(6):
            workspace.log_scalar(step, "eval/episode_return", float(step))
        workflow.graph.save()
        workflow.set_llm(
            MetricsAnalysisFake(
                messages=iter(()),
                evidence_ids=[
                    "eval_return.window_initial.mean",
                    "eval_return.window_final.mean",
                ],
            )
        )
        handler = REGISTRY.get("subagent_metrics_analysis").build_handler(workflow)

        handler({})

        event = json.loads(audit_index_path(prepared_experiment).read_text(encoding="utf-8").splitlines()[0])
        child_thread = f"{workflow.thread_id}:metrics_analysis"
        child_root = subagent_run_dir(prepared_experiment, workflow.thread_id, "metrics_analysis")
        manifest = json.loads((child_root / "manifest.json").read_text(encoding="utf-8"))

        assert event["tool"] == "subagent_metrics_analysis"
        assert event["kind"] == "subagent"
        assert event["thread_id"] == workflow.thread_id
        assert event["sub_thread_id"] == child_thread
        assert event["parent_tool"] == "subagent_metrics_analysis"
        assert event["run_dir"].endswith("/subagents/metrics_analysis")
        assert (prepared_experiment / event["run_dir"] / "manifest.json").is_file()
        assert manifest["parent_thread_id"] == workflow.thread_id
        assert manifest["thread_id"] == child_thread
