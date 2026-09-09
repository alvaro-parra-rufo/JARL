"""Tests for ``jarl.training.cli``."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from jarl.agents.ppo.gru_trainer import ppo_gru_full_jax_trainer
from jarl.experiments.graph import ExperimentGraph
from jarl.training.cli import (
    _install_sigterm_interrupt,
    load_config_snapshot,
    main,
    parse_args,
    resolve_trainer_for_node,
)
from jarl.training.launch import TRAIN_WORKER_PID_NAME
from jarl.training.run_overrides import mutable_overrides_from_config
from tests.helpers.fake_trainer import fake_trainer
from tests.helpers.minimal_navix_configs import minimal_ppo_gru_run_config, minimal_ppo_run_config


class TestTrainingCliConfigSnapshot:
    """JSON snapshot loading and trainer resolution."""

    def test_load_config_snapshot(self, tmp_path: Path) -> None:
        config = minimal_ppo_run_config()
        path = tmp_path / "pending.json"
        path.write_text(json.dumps(config.model_dump()), encoding="utf-8")

        loaded = load_config_snapshot(path)

        assert loaded.algorithm.name == config.algorithm.name

    def test_mutable_overrides_from_snapshot(self) -> None:
        config = minimal_ppo_run_config().apply_overrides(
            {
                "tracking.track_wandb": True,
                "tracking.wandb_project": "jarl-lab",
            }
        )

        overrides = mutable_overrides_from_config(config)

        assert overrides["tracking.track_wandb"] is True
        assert overrides["tracking.wandb_project"] == "jarl-lab"

    def test_resolve_trainer_for_node_uses_lineage_not_fallback(self, tmp_path: Path) -> None:
        gru_config = minimal_ppo_gru_run_config()
        exp_dir = tmp_path / "exp"
        graph = ExperimentGraph(exp_dir, base_config=gru_config)
        parent = graph.create_root(config=gru_config, label="parent")
        child = graph.fork("child_branch", from_node=parent, prepare=True)
        graph.save()

        feedforward_config = minimal_ppo_run_config()
        resolved = resolve_trainer_for_node(
            exp_dir,
            node_id=child.id,
            fallback_config=feedforward_config,
        )

        assert resolved is ppo_gru_full_jax_trainer
        assert graph.resolve_config(child).algorithm.name == "ppo_gru.full_jax.navix"


class TestTrainingCliArgs:
    """Flag parsing for unified training entrypoint."""

    def test_parse_args_config_json_and_resume(self, tmp_path: Path) -> None:
        args = parse_args(
            [
                "--experiment-dir",
                str(tmp_path / "exp"),
                "--config-json",
                str(tmp_path / "config.json"),
                "--resume",
                "--node-id",
                "node_a",
            ]
        )

        assert args.config_json == str(tmp_path / "config.json")
        assert args.resume is True
        assert args.node_id == "node_a"


class TestTrainingCliMain:
    """``main()`` with ``--config-json`` delegates to runner helpers."""

    def test_main_train_with_config_json_on_existing_node(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        config = minimal_ppo_run_config().apply_overrides({"algorithm.learning_rate": 1e-5})
        exp_dir = tmp_path / "exp"
        graph = ExperimentGraph(exp_dir, base_config=config)
        graph.create_root(config=config, label="root")
        graph.save()
        node_id = graph.current_node.id
        config_path = tmp_path / "pending.json"
        config_path.write_text(json.dumps(config.model_dump()), encoding="utf-8")
        expected_overrides = mutable_overrides_from_config(config)

        mock_run_training = mocker.patch("jarl.training.cli.run_training")
        mocker.patch("jarl.training.cli.resolve_trainer_for_node", return_value=fake_trainer)

        exit_code = main(
            [
                "--experiment-dir",
                str(exp_dir),
                "--config-json",
                str(config_path),
                "--node-id",
                node_id,
            ]
        )

        assert exit_code == 0
        mock_run_training.assert_called_once_with(
            trainer=fake_trainer,
            experiment_dir=exp_dir,
            config=None,
            config_overrides=expected_overrides,
            create_root=False,
            node=node_id,
            label="",
        )

    def test_main_clears_matching_worker_pidfile(self, mocker: MockerFixture, tmp_path: Path) -> None:
        config = minimal_ppo_run_config()
        exp_dir = tmp_path / "exp"
        graph = ExperimentGraph(exp_dir, base_config=config)
        graph.create_root(config=config, label="root")
        graph.save()
        config_path = tmp_path / "pending.json"
        config_path.write_text(json.dumps(config.model_dump()), encoding="utf-8")
        pid_path = exp_dir / TRAIN_WORKER_PID_NAME
        pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        mocker.patch("jarl.training.cli.run_training")
        mocker.patch("jarl.training.cli.resolve_trainer_for_node", return_value=fake_trainer)

        exit_code = main(
            [
                "--experiment-dir",
                str(exp_dir),
                "--config-json",
                str(config_path),
                "--node-id",
                graph.current_node.id,
            ]
        )

        assert exit_code == 0
        assert not pid_path.exists()

    def test_main_create_root_with_config_json(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        config = minimal_ppo_run_config()
        exp_dir = tmp_path / "exp"
        config_path = tmp_path / "root.json"
        config_path.write_text(json.dumps(config.model_dump()), encoding="utf-8")

        mock_run_training = mocker.patch("jarl.training.cli.run_training")
        mocker.patch("jarl.training.cli.resolve_trainer_for_node", return_value=fake_trainer)

        exit_code = main(
            [
                "--experiment-dir",
                str(exp_dir),
                "--config-json",
                str(config_path),
                "--create-root",
                "--label",
                "baseline",
            ]
        )

        assert exit_code == 0
        mock_run_training.assert_called_once_with(
            trainer=fake_trainer,
            experiment_dir=exp_dir,
            config=config,
            config_overrides=None,
            create_root=True,
            node=None,
            label="baseline",
        )

    def test_main_resume_with_config_json(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        config = minimal_ppo_run_config()
        exp_dir = tmp_path / "exp"
        graph = ExperimentGraph(exp_dir, base_config=config)
        graph.create_root(config=config, label="root")
        graph.save()
        node_id = graph.current_node.id
        config_path = tmp_path / "resume.json"
        config_path.write_text(json.dumps(config.model_dump()), encoding="utf-8")
        expected_overrides = mutable_overrides_from_config(config)

        mock_resume_training = mocker.patch("jarl.training.cli.resume_training")
        mocker.patch("jarl.training.cli.resolve_trainer_for_node", return_value=fake_trainer)

        exit_code = main(
            [
                "--experiment-dir",
                str(exp_dir),
                "--config-json",
                str(config_path),
                "--resume",
                "--node-id",
                node_id,
            ]
        )

        assert exit_code == 0
        mock_resume_training.assert_called_once_with(
            trainer=fake_trainer,
            experiment_dir=exp_dir,
            node=node_id,
            config_overrides=expected_overrides,
        )

    def test_main_resume_requires_node_id(self, tmp_path: Path) -> None:
        config_path = tmp_path / "resume.json"
        config_path.write_text(json.dumps(minimal_ppo_run_config().model_dump()), encoding="utf-8")

        with pytest.raises(SystemExit, match="--node-id is required"):
            main(
                [
                    "--experiment-dir",
                    str(tmp_path / "exp"),
                    "--config-json",
                    str(config_path),
                    "--resume",
                ]
            )

    def test_sigterm_maps_to_keyboard_interrupt(self) -> None:
        previous = signal.getsignal(signal.SIGTERM)
        try:
            _install_sigterm_interrupt()
            handler = signal.getsignal(signal.SIGTERM)
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGTERM, None)
        finally:
            signal.signal(signal.SIGTERM, previous)
