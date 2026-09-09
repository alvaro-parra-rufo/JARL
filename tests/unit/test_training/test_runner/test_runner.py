"""Tests for the global RL training runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.layout import ExperimentLayout
from jarl.experiments.io.metrics import JsonlMetricReader
from jarl.experiments.manifest import MANIFEST_FILENAME
from jarl.experiments.node import NodeStatus, NodeWorkspace
from jarl.training.config import AlgorithmConfig, EnvironmentConfig, RLRunConfig, VideoConfig
from jarl.training.runner import resume_training, run_training
from jarl.training.schedule import TrainingSchedule
from jarl.training.trainer import EnvFactory
from tests.helpers.fake_trainer import fake_trainer


@pytest.fixture()
def runner_config() -> RLRunConfig:
    """Small RL config for fast runner lifecycle tests without real trainers."""
    return RLRunConfig(
        environment=EnvironmentConfig(nr_envs=4, seed=0),
        algorithm=AlgorithmConfig(
            total_timesteps=512,
            nr_steps=8,
            minibatch_size=32,
            nr_epochs=1,
            evaluation_and_save_frequency=-1,
            evaluation_active=False,
        ),
        video=VideoConfig(record_video=False, record_final_video=False),
    )


class TestRunTraining:
    """Runner orchestration with injected trainers (no compiled JAX loops)."""

    def test_create_root_run_persists_schedule_and_metrics(
        self,
        tmp_path: Path,
        runner_config: RLRunConfig,
    ) -> None:
        exp_dir = tmp_path / "exp"

        result = run_training(
            trainer=fake_trainer,
            experiment_dir=exp_dir,
            config=runner_config,
            create_root=True,
            label="baseline",
        )

        saved_config = json.loads(result.workspace.config_path.read_text(encoding="utf-8"))
        latest_metric = JsonlMetricReader(result.workspace.metrics_jsonl_path).latest()

        assert result.workspace.status == NodeStatus.COMPLETED
        assert saved_config["algorithm"]["actual_total_timesteps"] == result.schedule.actual_total_timesteps
        assert latest_metric["runner_smoke"] == pytest.approx(float(result.schedule.actual_rollout_updates))
        assert (exp_dir / MANIFEST_FILENAME).is_file()
        assert ExperimentLayout(exp_dir).manifest_path.is_file()

    def test_fork_override_run_child(self, tmp_path: Path, runner_config: RLRunConfig) -> None:
        exp_dir = tmp_path / "exp"
        root_result = run_training(
            trainer=fake_trainer,
            experiment_dir=exp_dir,
            config=runner_config,
            create_root=True,
        )
        graph = root_result.graph
        child_config = runner_config.apply_overrides({"algorithm.learning_rate": 1e-5})
        child = graph.fork("lr_exp", from_node=root_result.workspace, config=child_config)

        child_result = run_training(
            trainer=fake_trainer,
            experiment_dir=exp_dir,
            graph=graph,
            node=child,
        )

        saved_config = json.loads(child_result.workspace.config_path.read_text(encoding="utf-8"))

        assert child_result.workspace.status == NodeStatus.COMPLETED
        assert saved_config["algorithm"]["learning_rate"] == 1e-5
        assert saved_config["algorithm"]["actual_total_timesteps"] == child_result.schedule.actual_total_timesteps

    def test_resume_after_simulated_failure(self, tmp_path: Path, runner_config: RLRunConfig) -> None:
        exp_dir = tmp_path / "exp"

        def failing_trainer(
            workspace: NodeWorkspace,
            config: RLRunConfig,
            schedule: TrainingSchedule,
            *,
            env_factory: EnvFactory | None = None,
        ) -> None:
            del config, schedule, env_factory
            workspace.log_scalar(0, "attempt", 1.0)
            msg = "boom"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="boom"):
            run_training(
                trainer=failing_trainer,
                experiment_dir=exp_dir,
                config=runner_config,
                create_root=True,
            )

        graph = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
        failed_workspace = graph.current_node
        assert failed_workspace.status == NodeStatus.FAILED

        resumed = resume_training(
            trainer=fake_trainer,
            experiment_dir=exp_dir,
            node=failed_workspace,
        )

        assert resumed.workspace.status == NodeStatus.COMPLETED
        assert JsonlMetricReader(resumed.workspace.metrics_jsonl_path).latest()["runner_smoke"] == pytest.approx(
            float(resumed.schedule.actual_rollout_updates)
        )


class TestRunnerCli:
    """Tests for the thin CLI wrapper."""

    def test_build_config_overrides_from_flags(self, tmp_path: Path) -> None:
        from jarl.training.cli import build_config_overrides, parse_args

        args = parse_args(
            [
                "--experiment-dir",
                str(tmp_path / "exp"),
                "--algorithm-name",
                "ppo_gru.full_jax.navix",
                "--total-timesteps",
                "4096",
                "--nr-envs",
                "16",
                "--obs-encoding-dim",
                "32",
                "--gru-hidden-dim",
                "16",
                "--no-wandb",
            ]
        )

        overrides = build_config_overrides(args)

        assert overrides == {
            "algorithm.name": "ppo_gru.full_jax.navix",
            "algorithm.total_timesteps": 4096,
            "environment.nr_envs": 16,
            "algorithm.obs_encoding_dim": 32,
            "algorithm.gru_hidden_dim": 16,
            "tracking.track_wandb": False,
        }
