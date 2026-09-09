"""Tests for train operations."""

from __future__ import annotations

import pytest
from pytest_mock import MockerFixture

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.metrics import JsonlMetricReader
from jarl.experiments.node import NodeStatus, NodeWorkspace
from jarl.operations.train.resume import ResumeRequest, resume
from jarl.operations.train.run import RunRequest, run
from jarl.training.config import RLRunConfig
from jarl.training.launch import SpawnedTraining
from jarl.training.schedule import TrainingSchedule
from jarl.training.trainer import EnvFactory
from tests.helpers.fake_trainer import fake_trainer


class TestTrainRun:
    def test_run_on_prepared_node(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        response = run(
            prepared_root,
            RunRequest(node_id=prepared_root.current_node.id),
            trainer=fake_trainer,
        )

        assert response.status == NodeStatus.COMPLETED.value
        metric = JsonlMetricReader(prepared_root.current_node.metrics_jsonl_path).latest()
        assert metric["runner_smoke"] == pytest.approx(float(response.schedule.actual_rollout_updates))

    def test_run_rejects_immutable_config_override(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        with pytest.raises(ValueError, match="immutable paths"):
            run(
                prepared_root,
                RunRequest(
                    node_id=prepared_root.current_node.id,
                    config_overrides={"algorithm.name": "ppo.full_jax.navix"},
                ),
                trainer=fake_trainer,
            )

    def test_run_requires_trainer_unless_detached(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        with pytest.raises(ValueError, match="trainer is required"):
            run(prepared_root, RunRequest())

    def test_run_detached_spawns_without_trainer(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
        mocker: MockerFixture,
    ) -> None:
        log_path = prepared_root.layout.root / "runner_lab.log"
        spawned = SpawnedTraining(
            pid=42,
            log_path=log_path,
            config_path=prepared_root.layout.root / "pending.json",
        )
        mock_spawn = mocker.patch("jarl.operations.train.run.spawn_training", return_value=spawned)

        response = run(prepared_root, RunRequest(detach=True))

        assert response.detached is True
        assert response.pid == 42
        assert response.log_path == log_path.as_posix()
        assert response.status == NodeStatus.PREPARED.value
        command = mock_spawn.call_args.kwargs["command"]
        assert "--node-id" in command
        assert prepared_root.current_node.id in command


class TestTrainResume:
    def test_resume_failed_node(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
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
            run(prepared_root, RunRequest(), trainer=failing_trainer)

        response = resume(prepared_root, ResumeRequest(), trainer=fake_trainer)

        assert response.status == NodeStatus.COMPLETED.value

    def test_resume_rejects_completed_node(self, prepared_root: ExperimentGraph[RLRunConfig]) -> None:
        run(prepared_root, RunRequest(), trainer=fake_trainer)

        with pytest.raises(RuntimeError, match="not resumable"):
            resume(prepared_root, ResumeRequest(), trainer=fake_trainer)

    def test_resume_detached_spawns_without_trainer(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
        mocker: MockerFixture,
    ) -> None:
        node = prepared_root.current_node
        node._meta.status = NodeStatus.FAILED
        node._save_metadata()
        log_path = prepared_root.layout.root / "runner_lab.log"
        spawned = SpawnedTraining(
            pid=7,
            log_path=log_path,
            config_path=prepared_root.layout.root / "pending.json",
        )
        mock_spawn = mocker.patch("jarl.operations.train.resume.spawn_training", return_value=spawned)

        response = resume(prepared_root, ResumeRequest(detach=True))

        assert response.detached is True
        assert response.pid == 7
        command = mock_spawn.call_args.kwargs["command"]
        assert "--resume" in command
        assert node.id in command
