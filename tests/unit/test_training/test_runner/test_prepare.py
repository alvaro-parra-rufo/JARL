"""Tests for deferred node preparation and runner start."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeStatus, NodeWorkspace
from jarl.training.config import AlgorithmConfig, EnvironmentConfig, RLRunConfig
from jarl.training.runner import run_training
from jarl.training.trainer import Trainer
from tests.helpers.fake_trainer import fake_trainer


@pytest.fixture()
def smoke_config() -> RLRunConfig:
    """Small RL config for prepare/runner tests."""
    return RLRunConfig(
        environment=EnvironmentConfig(nr_envs=8, seed=7),
        algorithm=AlgorithmConfig(
            total_timesteps=8192,
            nr_steps=128,
            minibatch_size=1024,
            evaluation_and_save_frequency=-1,
        ),
    )


def _assert_prepare_layout_only(workspace: NodeWorkspace) -> None:
    assert workspace.status == NodeStatus.PREPARED
    assert workspace.metadata_path.is_file()
    assert workspace.config_path.is_file()
    assert not workspace.metrics_jsonl_path.exists()
    assert not workspace.layout.execution_attempts_path.exists()
    assert not workspace.log_path.exists()


class TestPrepareCreationContract:
    """``prepare=True`` on graph APIs persists layout without training side effects."""

    def test_create_root_prepare_persists_without_run_artifacts(
        self,
        tmp_path: Path,
        smoke_config: RLRunConfig,
    ) -> None:
        graph: ExperimentGraph[RLRunConfig] = ExperimentGraph(tmp_path / "exp", base_config=smoke_config)
        workspace = graph.create_root(config=smoke_config, branch="main", label="baseline", prepare=True)

        _assert_prepare_layout_only(workspace)

    def test_fork_prepare_persists_without_run_artifacts(
        self,
        tmp_path: Path,
        smoke_config: RLRunConfig,
    ) -> None:
        exp_dir = tmp_path / "exp"
        root_result = run_training(
            trainer=fake_trainer,
            experiment_dir=exp_dir,
            config=smoke_config,
            create_root=True,
        )
        child_config = smoke_config.apply_overrides({"algorithm.learning_rate": 1e-5})
        child = root_result.graph.fork(
            "lr_fork",
            from_node=root_result.workspace,
            config=child_config,
            prepare=True,
        )

        _assert_prepare_layout_only(child)

    def test_extend_prepare_persists_without_run_artifacts(
        self,
        tmp_path: Path,
        smoke_config: RLRunConfig,
    ) -> None:
        exp_dir = tmp_path / "exp"
        root_result = run_training(
            trainer=fake_trainer,
            experiment_dir=exp_dir,
            config=smoke_config,
            create_root=True,
        )
        child = root_result.graph.extend("main", label="child", prepare=True)

        _assert_prepare_layout_only(child)

    def test_prepare_reload_round_trip(
        self,
        tmp_path: Path,
        smoke_config: RLRunConfig,
    ) -> None:
        exp_dir = tmp_path / "exp"
        graph: ExperimentGraph[RLRunConfig] = ExperimentGraph(exp_dir, base_config=smoke_config)
        graph.create_root(config=smoke_config, branch="main", prepare=True)
        graph.save()

        restored = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)

        assert restored.current_node.status == NodeStatus.PREPARED

    def test_prepare_never_invokes_trainer(
        self,
        tmp_path: Path,
        smoke_config: RLRunConfig,
        mocker: MockerFixture,
    ) -> None:
        trainer = mocker.Mock(spec=Trainer)
        graph: ExperimentGraph[RLRunConfig] = ExperimentGraph(tmp_path / "exp", base_config=smoke_config)
        prepared = graph.create_root(config=smoke_config, branch="main", prepare=True)

        trainer.assert_not_called()

        run_training(
            trainer=trainer,
            experiment_dir=tmp_path / "exp",
            graph=graph,
            node=prepared,
        )

        trainer.assert_called_once()


class TestPreparedNodeRunnerIntegration:
    """Runner starts prepared nodes created via graph APIs."""

    @pytest.mark.parametrize(
        ("prepare_factory",),
        [
            pytest.param("create_root", id="create_root"),
            pytest.param("fork", id="fork"),
            pytest.param("extend", id="extend"),
        ],
    )
    def test_prepared_node_completes_via_runner(
        self,
        tmp_path: Path,
        smoke_config: RLRunConfig,
        prepare_factory: str,
    ) -> None:
        exp_dir = tmp_path / "exp"
        graph, prepared = _create_prepared_node(exp_dir, smoke_config, prepare_factory)

        result = run_training(
            trainer=fake_trainer,
            experiment_dir=exp_dir,
            graph=graph,
            node=prepared,
        )

        assert result.workspace.status == NodeStatus.COMPLETED
        assert result.workspace.metrics_jsonl_path.is_file()


def _create_prepared_node(
    exp_dir: Path,
    smoke_config: RLRunConfig,
    factory: str,
) -> tuple[ExperimentGraph[RLRunConfig], NodeWorkspace]:
    if factory == "create_root":
        graph: ExperimentGraph[RLRunConfig] = ExperimentGraph(exp_dir, base_config=smoke_config)
        prepared = graph.create_root(config=smoke_config, branch="main", prepare=True)
        return graph, prepared

    root_result = run_training(
        trainer=fake_trainer,
        experiment_dir=exp_dir,
        config=smoke_config,
        create_root=True,
    )
    graph = root_result.graph
    if factory == "fork":
        child_config = smoke_config.apply_overrides({"algorithm.learning_rate": 1e-5})
        prepared = graph.fork("lr_fork", from_node=root_result.workspace, config=child_config, prepare=True)
    elif factory == "extend":
        prepared = graph.extend("main", label="child", prepare=True)
    else:
        msg = f"Unknown prepare factory: {factory!r}"
        raise ValueError(msg)
    return graph, prepared


class TestRunTrainingNodeSelection:
    """Runner executes existing nodes only."""

    def test_empty_experiment_requires_create_root(
        self,
        tmp_path: Path,
        smoke_config: RLRunConfig,
    ) -> None:
        with pytest.raises(RuntimeError, match="Experiment has no nodes"):
            run_training(
                trainer=fake_trainer,
                experiment_dir=tmp_path / "exp",
                config=smoke_config,
            )

    def test_create_root_trains_immediately_without_prepare_flag(
        self,
        tmp_path: Path,
        smoke_config: RLRunConfig,
        mocker: MockerFixture,
    ) -> None:
        setup_jax = mocker.patch("jarl.training.runner.setup_jax")

        result = run_training(
            trainer=fake_trainer,
            experiment_dir=tmp_path / "exp",
            config=smoke_config,
            create_root=True,
        )

        setup_jax.assert_called_once()
        assert result.workspace.status == NodeStatus.COMPLETED
