"""Tests for train agentic tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from jarl.agentic.audit import audit_index_path
from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.registry import REGISTRY
from jarl.agentic.tools.train import resume as train_resume_module
from jarl.agentic.tools.train import run as train_run_module
from jarl.agentic.tools.train._form import (
    RunFormPayloadRequest,
    RunFormValuesRequest,
    build_run_form_payload,
)
from jarl.agentic.tools.train.resume import ToolRequest as ResumeToolRequest
from jarl.agentic.tools.train.resume import run_train_resume
from jarl.agentic.tools.train.run import ToolRequest as RunToolRequest
from jarl.agentic.tools.train.run import run_train_run
from jarl.agentic.workflow import AgenticWorkflow
from jarl.experiments.io.metrics import JsonlMetricReader
from jarl.experiments.node import NodeStatus, NodeWorkspace
from jarl.operations.train.run import RunRequest, run
from jarl.training.config import RLRunConfig
from jarl.training.launch import TRAIN_DETACH_ENV, SpawnedTraining
from jarl.training.presets import form_values_to_run_config
from jarl.training.schedule import TrainingSchedule
from jarl.training.trainer import EnvFactory
from tests.helpers.fake_trainer import fake_trainer


@pytest.fixture(autouse=True)
def fake_trainer_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch trainer resolution to the smoke fake trainer."""

    def _resolve(_ctx: ToolContext, *, node_id: str | None) -> object:
        del node_id
        return fake_trainer

    monkeypatch.setattr(train_run_module, "resolve_trainer", _resolve)
    monkeypatch.setattr(train_resume_module, "resolve_trainer", _resolve)


class TestTrainRunTool:
    def test_run_on_prepared_node(self, tool_context: ToolContext) -> None:
        payload = json.loads(run_train_run(tool_context, RunToolRequest()))

        assert payload["status"] == NodeStatus.COMPLETED.value
        metric = JsonlMetricReader(tool_context.graph.current_node.metrics_jsonl_path).latest()
        assert metric["runner_smoke"] == pytest.approx(float(payload["schedule"]["actual_rollout_updates"]))

    def test_run_form_with_zero_video_frequency_builds_valid_config(self) -> None:
        payload = build_run_form_payload(
            RunFormPayloadRequest(
                values=RunFormValuesRequest(preset="fast"),
                video_frequency=0,
            )
        )
        assert payload is not None
        config = form_values_to_run_config(payload)
        assert config.video.video_frequency >= 1

    def test_run_detaches_when_env_set(self, tool_context: ToolContext, mocker: MockerFixture) -> None:
        mocker.patch.dict("os.environ", {TRAIN_DETACH_ENV: "1"}, clear=False)
        log_path = tool_context.exp_dir / "runner_lab.log"
        spawned = SpawnedTraining(
            pid=11,
            log_path=log_path,
            config_path=tool_context.exp_dir / "pending.json",
        )
        mock_spawn = mocker.patch("jarl.operations.train.run.spawn_training", return_value=spawned)

        payload = json.loads(run_train_run(tool_context, RunToolRequest()))

        assert payload["detached"] is True
        assert payload["pid"] == 11
        mock_spawn.assert_called_once()


class TestTrainResumeTool:
    def test_resume_failed_node(self, tool_context: ToolContext) -> None:
        graph = tool_context.graph

        def failing_trainer(
            _workspace: NodeWorkspace,
            config: RLRunConfig,
            schedule: TrainingSchedule,
            *,
            env_factory: EnvFactory | None = None,
        ) -> None:
            del config, schedule, env_factory
            msg = "boom"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="boom"):
            run(graph, RunRequest(), trainer=failing_trainer)

        payload = json.loads(run_train_resume(tool_context, ResumeToolRequest()))

        assert payload["status"] == NodeStatus.COMPLETED.value

    def test_resume_rejects_completed_node(self, tool_context: ToolContext) -> None:
        run_train_run(tool_context, RunToolRequest())

        with pytest.raises(RuntimeError, match="not resumable"):
            run_train_resume(tool_context, ResumeToolRequest())


class TestTrainAudit:
    def test_train_run_audits_via_wire_handler(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment, audit_reads=False)
        handler = REGISTRY.get("train_run").build_handler(workflow)

        handler({})

        event = json.loads(audit_index_path(prepared_experiment).read_text(encoding="utf-8").splitlines()[0])
        assert event["tool"] == "train_run"
        assert event["kind"] == "train"

    def test_train_resume_audits_via_wire_handler(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment, audit_reads=False)
        root_id = workflow.current_node_id

        def failing_trainer(
            _workspace: NodeWorkspace,
            config: RLRunConfig,
            schedule: TrainingSchedule,
            *,
            env_factory: EnvFactory | None = None,
        ) -> None:
            del config, schedule, env_factory
            msg = "boom"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="boom"):
            run(
                workflow.graph,
                RunRequest(),
                trainer=failing_trainer,
            )

        resume_handler = REGISTRY.get("train_resume").build_handler(workflow)
        resume_handler({})

        events = [
            json.loads(line) for line in audit_index_path(prepared_experiment).read_text(encoding="utf-8").splitlines()
        ]
        resume_event = next(event for event in events if event["tool"] == "train_resume")
        assert resume_event["kind"] == "train"
        assert resume_event["node_before"] == root_id
