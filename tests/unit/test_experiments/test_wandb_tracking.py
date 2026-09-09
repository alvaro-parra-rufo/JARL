"""Tests for optional Weights & Biases tracking."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from jarl.experiments.io.metrics import JsonlMetricReader
from jarl.experiments.node import NodeMetadata, NodeWorkspace
from jarl.experiments.run_config import TrackingConfig
from jarl.experiments.wandb_tracking import (
    DEFAULT_WANDB_PROJECT,
    WandbRunTracker,
    build_wandb_run_config,
    log_wandb_scalar,
)
from jarl.training.config import AlgorithmConfig, EnvironmentConfig, RLRunConfig
from jarl.training.schedule import compute_training_schedule


class TestLogWandbScalar:
    """Tests for scalar logging helper."""

    def test_log_wandb_scalar_uses_explicit_step_when_monotonic(self, mocker: MockerFixture) -> None:
        mock_run = mocker.Mock(step=100)
        mocker.patch("wandb.run", new=mock_run)
        mock_log = mocker.patch("wandb.log")

        log_wandb_scalar(100, "loss", 0.5)

        mock_log.assert_called_once_with({"loss": 0.5, "global_step": 100}, step=100)


class TestWandbRunTracker:
    """Unit tests for the W&B tracker helper."""

    def test_init_skipped_when_tracking_disabled(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mock_init = mocker.patch("wandb.init")

        tracker = WandbRunTracker(
            tracking=TrackingConfig(track_wandb=False),
            wandb_dir=tmp_path / "wandb",
            run_name="main_test_ab12cd34",
        )
        tracker.init()

        mock_init.assert_not_called()
        assert not tracker.active

    def test_init_creates_wandb_dir_and_configures_run(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mock_init = mocker.patch("wandb.init")
        mock_define = mocker.patch("jarl.experiments.wandb_tracking.configure_wandb_metrics")
        wandb_dir = tmp_path / "wandb"

        tracker = WandbRunTracker(
            tracking=TrackingConfig(
                track_wandb=True,
                wandb_project="my-project",
                wandb_group="exp-a",
                wandb_tags=["tag-one"],
                wandb_mode="offline",
            ),
            wandb_dir=wandb_dir,
            run_name="main_test_ab12cd34",
            config={"node_id": "main_test_ab12cd34"},
        )
        tracker.init()

        mock_init.assert_called_once_with(
            project="my-project",
            name="main_test_ab12cd34",
            dir=str(wandb_dir),
            mode="offline",
            config={"node_id": "main_test_ab12cd34"},
            group="exp-a",
            tags=["tag-one"],
        )
        mock_define.assert_called_once()
        assert wandb_dir.is_dir()
        assert tracker.active

    def test_init_uses_wandb_mode_from_environment_when_unset(
        self, tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WANDB_MODE", "online")
        mock_init = mocker.patch("wandb.init")
        mocker.patch("jarl.experiments.wandb_tracking.configure_wandb_metrics")

        tracker = WandbRunTracker(
            tracking=TrackingConfig(track_wandb=True),
            wandb_dir=tmp_path / "wandb",
            run_name="main_test_ab12cd34",
        )
        tracker.init()

        assert mock_init.call_args.kwargs["mode"] == "online"

    def test_init_failure_disables_tracking_without_raising(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch("wandb.init", side_effect=RuntimeError("network down"))

        tracker = WandbRunTracker(
            tracking=TrackingConfig(track_wandb=True),
            wandb_dir=tmp_path / "wandb",
            run_name="main_test_ab12cd34",
        )
        tracker.init()

        assert not tracker.active
        tracker.log_scalar(1, "loss", 0.5)

    def test_finish_closes_active_run(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch("wandb.init")
        mocker.patch("jarl.experiments.wandb_tracking.configure_wandb_metrics")
        mock_finish = mocker.patch("wandb.finish")
        mocker.patch("wandb.run", new=mocker.Mock())

        tracker = WandbRunTracker(
            tracking=TrackingConfig(track_wandb=True),
            wandb_dir=tmp_path / "wandb",
            run_name="main_test_ab12cd34",
        )
        tracker.init()
        tracker.finish()

        mock_finish.assert_called_once()


class TestBuildWandbRunConfig:
    """Tests for enriched W&B config payloads."""

    def test_includes_resolved_config_schedule_and_paths(self, tmp_path: Path) -> None:
        metadata = NodeMetadata(id="main_test_ab12cd34", branch="main", label="baseline")
        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=metadata,
        )
        config = RLRunConfig(
            environment=EnvironmentConfig(env_id="Navix-Empty-5x5-v0", nr_envs=4, seed=7),
            algorithm=AlgorithmConfig(
                name="ppo.full_jax.navix",
                total_timesteps=512,
                nr_steps=8,
                minibatch_size=32,
            ),
        )
        schedule = compute_training_schedule(config.environment, config.algorithm)
        payload = build_wandb_run_config(
            node_metadata=metadata,
            node_layout=workspace.layout,
            config=config,
            schedule=schedule,
        )

        assert payload["node_id"] == "main_test_ab12cd34"
        assert payload["algorithm_name"] == "ppo.full_jax.navix"
        assert payload["env_id"] == "Navix-Empty-5x5-v0"
        assert payload["actual_total_timesteps"] == schedule.actual_total_timesteps
        assert payload["batch_size"] == schedule.batch_size
        assert payload["config_path"].endswith("config.json")
        assert payload["metrics_path"].endswith("metrics.jsonl")

    def test_bind_wandb_run_context_forwards_payload_to_init(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mock_init = mocker.patch("wandb.init")
        mocker.patch("jarl.experiments.wandb_tracking.configure_wandb_metrics")

        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_test_ab12cd34", branch="main"),
            tracking=TrackingConfig(track_wandb=True),
        )
        config = RLRunConfig(
            environment=EnvironmentConfig(env_id="Navix-Empty-5x5-v0", nr_envs=4),
            algorithm=AlgorithmConfig(total_timesteps=512, nr_steps=8, minibatch_size=32),
        )
        schedule = compute_training_schedule(config.environment, config.algorithm)
        workspace.bind_wandb_run_context(config, schedule)

        with workspace:
            pass

        init_config = mock_init.call_args.kwargs["config"]
        assert init_config["env_id"] == "Navix-Empty-5x5-v0"
        assert init_config["actual_rollout_updates"] == schedule.actual_rollout_updates


class TestNodeWorkspaceWandbIntegration:
    """Tests for W&B integration in the node training lifecycle."""

    @pytest.fixture()
    def workspace(self, tmp_path: Path) -> NodeWorkspace:
        """Workspace with W&B disabled by default."""
        return NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_test_ab12cd34"),
        )

    def test_wandb_off_has_no_side_effects(self, workspace: NodeWorkspace, mocker: MockerFixture) -> None:
        mock_init = mocker.patch("wandb.init")

        with workspace:
            workspace.log_scalar(0, "loss", 0.4)

        mock_init.assert_not_called()
        assert not workspace.wandb_dir.exists()
        assert not workspace.wandb_active

    def test_wandb_on_initializes_and_finishes(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mock_init = mocker.patch("wandb.init")
        mock_finish = mocker.patch("wandb.finish")
        mocker.patch("wandb.run", new=mocker.Mock())
        mocker.patch("jarl.experiments.wandb_tracking.configure_wandb_metrics")
        mock_log = mocker.patch("jarl.experiments.wandb_tracking.log_wandb_scalar")

        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_test_ab12cd34", branch="main"),
            tracking=TrackingConfig(track_wandb=True, wandb_project=""),
        )

        with workspace:
            workspace.log_scalar(2, "accuracy", 0.9)

        mock_init.assert_called_once()
        init_kwargs = mock_init.call_args.kwargs
        assert init_kwargs["project"] == DEFAULT_WANDB_PROJECT
        assert init_kwargs["name"] == "main_test_ab12cd34"
        assert Path(init_kwargs["dir"]).name == "wandb"
        mock_log.assert_called_once_with(2, "accuracy", 0.9)
        mock_finish.assert_called_once()
        assert workspace.wandb_dir.is_dir()
        assert JsonlMetricReader(workspace.metrics_jsonl_path).latest() == {"accuracy": 0.9}

    def test_init_failure_allows_training_to_continue(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch("wandb.init", side_effect=RuntimeError("init failed"))
        mocker.patch("wandb.finish")

        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_test_ab12cd34"),
            tracking=TrackingConfig(track_wandb=True),
        )

        with workspace:
            workspace.log_scalar(1, "loss", 0.2)

        assert not workspace.wandb_active
        assert JsonlMetricReader(workspace.metrics_jsonl_path).latest() == {"loss": 0.2}

    def test_log_scalars_forwards_to_wandb(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch("wandb.init")
        mocker.patch("wandb.finish")
        mocker.patch("wandb.run", new=mocker.Mock())
        mocker.patch("jarl.experiments.wandb_tracking.configure_wandb_metrics")
        mock_log = mocker.patch("jarl.experiments.wandb_tracking.log_wandb_scalar")

        workspace = NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_test_ab12cd34"),
            tracking=TrackingConfig(track_wandb=True),
        )

        with workspace:
            workspace.log_scalars(3, loss=0.2, accuracy=0.8)

        assert mock_log.call_count == 2
        mock_log.assert_any_call(3, "loss", 0.2)
        mock_log.assert_any_call(3, "accuracy", 0.8)
