"""Tests for TensorBoard experiment helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from jarl.config import BaseConfig
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.tensorboard import (
    build_logdir_spec,
    launch_tensorboard,
    tensorboard_compare,
    tensorboard_lineage,
)


class SampleConfig(BaseConfig):
    """Concrete config for tensorboard tests."""

    learning_rate: float = 0.1


@pytest.fixture()
def graph(tmp_path: Path) -> ExperimentGraph[SampleConfig]:
    """Graph with root and forked branch."""
    exp_dir = tmp_path / "exp"
    g: ExperimentGraph[SampleConfig] = ExperimentGraph(exp_dir)
    root = g.create_root(config=SampleConfig(), branch="main", label="baseline")
    g.fork("lr_exp", from_node=root, config=SampleConfig(learning_rate=0.01), label="higher_lr")
    return g


class TestBuildLogdirSpec:
    """Tests for logdir_spec formatting."""

    def test_formats_labeled_paths(self, tmp_path: Path) -> None:
        path_a = tmp_path / "a"
        path_b = tmp_path / "b"

        result = build_logdir_spec({"run_a": path_a, "run_b": path_b})

        assert result == f"run_a:{path_a.resolve().as_posix()},run_b:{path_b.resolve().as_posix()}"

    def test_empty_entries_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            build_logdir_spec({})


class TestTensorboardLineage:
    """Tests for lineage logdir specs."""

    def test_includes_all_lineage_nodes(self, graph: ExperimentGraph[SampleConfig]) -> None:
        fork_head = graph.head("lr_exp")

        spec = tensorboard_lineage(graph, fork_head)

        assert "0_main:" in spec
        assert "1_lr_exp:" in spec
        assert str(graph.head("main").tensorboard_dir.resolve().as_posix()) in spec

    def test_root_lineage_single_entry(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")

        spec = tensorboard_lineage(graph, root)

        assert spec.startswith("0_main:")
        assert spec.count(",") == 0


class TestTensorboardCompare:
    """Tests for compare logdir specs."""

    def test_compare_branches(self, graph: ExperimentGraph[SampleConfig]) -> None:
        spec = tensorboard_compare(graph, ["main", "lr_exp"])

        assert graph.head("main").id in spec
        assert graph.head("lr_exp").id in spec
        assert "," in spec

    def test_compare_by_node_id(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")

        spec = tensorboard_compare(graph, [root.id])

        assert root.id in spec

    def test_unknown_target_raises(self, graph: ExperimentGraph[SampleConfig]) -> None:
        with pytest.raises(KeyError, match="Unknown node or branch"):
            tensorboard_compare(graph, ["missing"])


class TestLaunchTensorboard:
    """Tests for TensorBoard subprocess launcher."""

    def test_launches_subprocess(self, mocker: MockerFixture) -> None:
        mocker.patch("jarl.experiments.tensorboard.shutil.which", return_value="/usr/bin/tensorboard")
        popen = mocker.patch("jarl.experiments.tensorboard.subprocess.Popen")
        popen.return_value.poll.return_value = None

        launch_tensorboard("main:/tmp/tb", port=6007)

        popen.assert_called_once()
        cmd = popen.call_args.args[0]
        assert cmd[0] == "/usr/bin/tensorboard"
        assert "--logdir_spec=main:/tmp/tb" in cmd
        assert "--port=6007" in cmd

    def test_immediate_exit_raises(self, mocker: MockerFixture) -> None:
        mocker.patch("jarl.experiments.tensorboard.shutil.which", return_value="/usr/bin/tensorboard")
        popen = mocker.patch("jarl.experiments.tensorboard.subprocess.Popen")
        proc = popen.return_value
        proc.poll.return_value = 1
        proc.returncode = 1
        proc.stderr.read.return_value = "startup failed"

        with pytest.raises(RuntimeError, match="TensorBoard exited immediately"):
            launch_tensorboard("main:/tmp/tb")

    def test_missing_executable_raises(self, mocker: MockerFixture) -> None:
        mocker.patch("jarl.experiments.tensorboard.shutil.which", return_value=None)

        with pytest.raises(FileNotFoundError, match="tensorboard executable"):
            launch_tensorboard("main:/tmp/tb")
