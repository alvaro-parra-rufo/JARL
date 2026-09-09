"""Tests for agentic audit and run tracking."""

from __future__ import annotations

import json
from pathlib import Path

from jarl.agentic.audit import (
    RUN_EVENTS_FILENAME,
    RUN_MANIFEST_FILENAME,
    AuditIndexEvent,
    append_audit_index,
    audit_index_path,
    load_audit_index,
    load_run_events,
    open_run,
    run_dir,
    subagent_run_dir,
)
from jarl.agentic.run_events import ExperimentInvokeEvent, GraphNodeEvent
from jarl.agentic.session import AGENTIC_SESSION_FILENAME, load_or_create_session, load_session
from jarl.experiments.manifest import MANIFEST_FILENAME

_ALLOWED_SESSION_KEYS = frozenset(
    {
        "thread_id",
        "created_at",
        "updated_at",
        "audit_reads",
        "compiled_nodes",
    }
)
_GRAPH_STATE_KEYS = frozenset(
    {
        "current_node",
        "branch_heads",
        "nodes",
        "edges",
        "node_count",
        "execution_state",
    }
)


class TestAuditIndex:
    def test_append_audit_index_writes_jsonl(self, tmp_path: Path) -> None:
        event = AuditIndexEvent(
            ts="2026-07-22T12:00:00+00:00",
            thread_id="thread_abc",
            tool="graph/summary",
            kind="read",
            request={"node_id": None},
            result={"node_count": 1},
        )

        append_audit_index(tmp_path, event)

        lines = audit_index_path(tmp_path).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["tool"] == "graph/summary"
        assert payload["kind"] == "read"
        assert "node_before" not in payload

    def test_load_audit_index_round_trips_events(self, tmp_path: Path) -> None:
        event = AuditIndexEvent(
            ts="2026-07-22T12:00:00+00:00",
            thread_id="thread_abc",
            tool="graph_fork",
            kind="mutation",
            request={"branch": "exp"},
            result={"node_id": "exp_123"},
            node_before="main_123",
            node_after="exp_123",
        )
        append_audit_index(tmp_path, event)

        loaded = load_audit_index(tmp_path)

        assert loaded == (event,)

    def test_load_audit_index_round_trips_subagent_link_fields(self, tmp_path: Path) -> None:
        event = AuditIndexEvent(
            ts="2026-07-22T12:00:00+00:00",
            thread_id="thread_parent",
            tool="subagent_metrics_analysis",
            kind="subagent",
            request={},
            result={"labels": ["improving"]},
            sub_thread_id="thread_parent:metrics_analysis",
            parent_tool="subagent_metrics_analysis",
            run_dir=".agentic/runs/thread_parent/subagents/metrics_analysis",
        )
        append_audit_index(tmp_path, event)

        loaded = load_audit_index(tmp_path)

        assert loaded == (event,)
        payload = json.loads(audit_index_path(tmp_path).read_text(encoding="utf-8").splitlines()[0])
        assert payload["sub_thread_id"] == "thread_parent:metrics_analysis"
        assert payload["parent_tool"] == "subagent_metrics_analysis"
        assert payload["run_dir"] == ".agentic/runs/thread_parent/subagents/metrics_analysis"

    def test_load_audit_index_returns_empty_when_missing(self, tmp_path: Path) -> None:
        assert load_audit_index(tmp_path) == ()


class TestLoadRunEvents:
    def test_returns_empty_when_file_missing(self, tmp_path: Path) -> None:
        assert load_run_events(tmp_path, "thread_missing") == ()

    def test_parses_event_lines(self, tmp_path: Path) -> None:
        tracker = open_run(tmp_path, "thread_events", graph_id="experiment")
        invoke = ExperimentInvokeEvent(phase="setup", message_count=1)
        node = GraphNodeEvent(node="operate", message_count=3)
        tracker.append_event(invoke)
        tracker.append_event(node)

        loaded = load_run_events(tmp_path, "thread_events")

        assert loaded == (invoke, node)

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        tracker = open_run(tmp_path, "thread_blank", graph_id="experiment")
        first = GraphNodeEvent(node="setup", message_count=1)
        second = GraphNodeEvent(node="operate", message_count=2)
        events_path = tracker.path / RUN_EVENTS_FILENAME
        events_path.write_text(
            first.model_dump_json(exclude_none=True) + "\n\n" + second.model_dump_json(exclude_none=True) + "\n",
            encoding="utf-8",
        )

        loaded = load_run_events(tmp_path, "thread_blank")

        assert loaded == (first, second)


class TestRunTracking:
    def test_open_run_creates_manifest_and_events(self, tmp_path: Path) -> None:
        tracker = open_run(
            tmp_path,
            "thread_main",
            run_kind="agent",
            graph_id="experiment",
        )

        tracker.append_event(GraphNodeEvent(node="operate", message_count=1))
        tracker.complete()

        root = run_dir(tmp_path, "thread_main")
        manifest = json.loads((root / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        events = (root / RUN_EVENTS_FILENAME).read_text(encoding="utf-8").splitlines()

        assert manifest["thread_id"] == "thread_main"
        assert manifest["status"] == "completed"
        assert manifest["run_kind"] == "agent"
        assert len(events) == 1

    def test_open_run_subagent_records_parent(self, tmp_path: Path) -> None:
        tracker = open_run(
            tmp_path,
            "thread_parent:metrics_analysis",
            parent_thread_id="thread_parent",
            run_kind="subagent",
            graph_id="subagents/metrics_analysis",
            parent_tool="subagent_metrics_analysis",
            thread_suffix="metrics_analysis",
        )

        assert tracker.manifest.parent_thread_id == "thread_parent"
        assert tracker.manifest.parent_tool == "subagent_metrics_analysis"
        assert tracker.path == subagent_run_dir(tmp_path, "thread_parent", "metrics_analysis")
        assert tracker.path.is_relative_to(run_dir(tmp_path, "thread_parent"))

    def test_open_run_records_llm_identity(self, tmp_path: Path) -> None:
        tracker = open_run(
            tmp_path,
            "thread_llm",
            run_kind="agent",
            graph_id="experiment",
            llm_profile="powerful",
            llm_provider="ollama",
            llm_model="qwen2.5:32b",
        )

        manifest = json.loads((tracker.path / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert manifest["llm_profile"] == "powerful"
        assert manifest["llm_provider"] == "ollama"
        assert manifest["llm_model"] == "qwen2.5:32b"

    def test_open_run_reuses_existing_manifest(self, tmp_path: Path) -> None:
        first = open_run(tmp_path, "thread_reuse", graph_id="experiment")
        first.append_event(GraphNodeEvent(node="operate", message_count=1))
        second = open_run(tmp_path, "thread_reuse", graph_id="experiment")

        events = (second.path / RUN_EVENTS_FILENAME).read_text(encoding="utf-8").splitlines()

        assert second.manifest.thread_id == first.thread_id
        assert len(events) == 1


class TestSessionPersistence:
    def test_load_session_after_workflow_bootstrap(self, tmp_path: Path) -> None:
        from jarl.agentic.workflow import AgenticWorkflow
        from jarl.experiments.graph import ExperimentGraph
        from jarl.training.config import RLRunConfig

        exp_dir = tmp_path / "exp"
        graph = ExperimentGraph(exp_dir, base_config=RLRunConfig())
        graph.create_root(RLRunConfig(), label="root", prepare=True)
        graph.save()

        workflow = AgenticWorkflow.from_experiment(exp_dir, audit_reads=True)
        session = load_session(exp_dir)

        assert session is not None
        assert session.thread_id == workflow.thread_id
        assert session.audit_reads is True
        assert (exp_dir / AGENTIC_SESSION_FILENAME).is_file()

    def test_load_or_create_session_updates_audit_reads(self, tmp_path: Path) -> None:
        from jarl.experiments.graph import ExperimentGraph
        from jarl.training.config import RLRunConfig

        exp_dir = tmp_path / "exp"
        graph = ExperimentGraph(exp_dir, base_config=RLRunConfig())
        graph.create_root(RLRunConfig(), label="root", prepare=True)
        graph.save()

        first = load_or_create_session(exp_dir)
        assert first.audit_reads is False

        updated = load_or_create_session(exp_dir, audit_reads=True)

        assert updated.audit_reads is True
        assert load_session(exp_dir) is not None
        assert load_session(exp_dir).audit_reads is True

    def test_agentic_session_file_excludes_graph_state(self, tmp_path: Path) -> None:
        from jarl.agentic.workflow import AgenticWorkflow
        from jarl.experiments.graph import ExperimentGraph
        from jarl.training.config import RLRunConfig

        exp_dir = tmp_path / "exp"
        graph = ExperimentGraph(exp_dir, base_config=RLRunConfig())
        graph.create_root(RLRunConfig(), label="root", prepare=True)
        graph.fork("alt", label="child", prepare=True)
        graph.save()

        workflow = AgenticWorkflow.from_experiment(exp_dir, audit_reads=True)
        session_payload = json.loads((exp_dir / AGENTIC_SESSION_FILENAME).read_text(encoding="utf-8"))
        manifest_payload = json.loads((exp_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))

        assert set(session_payload.keys()) == _ALLOWED_SESSION_KEYS
        assert _GRAPH_STATE_KEYS.isdisjoint(session_payload.keys())
        assert "current_node" in manifest_payload
        assert manifest_payload["current_node"] == workflow.current_node_id

    def test_from_experiment_updates_audit_reads_on_existing_session(self, tmp_path: Path) -> None:
        from jarl.agentic.workflow import AgenticWorkflow
        from jarl.experiments.graph import ExperimentGraph
        from jarl.training.config import RLRunConfig

        exp_dir = tmp_path / "exp"
        graph = ExperimentGraph(exp_dir, base_config=RLRunConfig())
        graph.create_root(RLRunConfig(), label="root", prepare=True)
        graph.save()

        first = AgenticWorkflow.from_experiment(exp_dir)
        assert first.session.audit_reads is False

        second = AgenticWorkflow.from_experiment(exp_dir, audit_reads=True)

        assert second.thread_id == first.thread_id
        assert second.session.audit_reads is True
        persisted = load_session(exp_dir)
        assert persisted is not None
        assert persisted.audit_reads is True
