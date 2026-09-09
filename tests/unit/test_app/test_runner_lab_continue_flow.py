"""Tests for continuation flow helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.app.lib.continue_flow import (
    CheckpointPick,
    ContinuationRequest,
    default_extend_label,
    default_fork_branch,
    run_continuation,
)
from jarl.app.lib.inspect_view import can_extend_from_head, config_diff_dataframe
from jarl.config import ConfigDiff
from jarl.experiments.graph import ExperimentGraph
from jarl.training.config import RLRunConfig
from tests.helpers.minimal_navix_configs import minimal_ppo_run_config


class TestContinuationRequest:
    def test_default_labels(self) -> None:
        class _Meta:
            step = 2
            label = "root"

        class _Workspace:
            branch = "main"
            node_metadata = _Meta()

        workspace = _Workspace()
        assert default_extend_label(workspace) == "extend_3"  # type: ignore[arg-type]
        assert default_fork_branch(workspace) == "main_fork"  # type: ignore[arg-type]

    def test_checkpoint_pick_values(self) -> None:
        assert CheckpointPick.LATEST.value == "latest"


class TestInspectView:
    def test_config_diff_dataframe_empty(self) -> None:
        frame = config_diff_dataframe(ConfigDiff(added={}, removed={}, changed={}))
        assert frame.empty

    def test_config_diff_dataframe_changed(self) -> None:
        diff = ConfigDiff(added={}, removed={}, changed={"algorithm.learning_rate": (1e-4, 1e-5)})
        frame = config_diff_dataframe(diff)
        assert len(frame) == 1
        assert frame.iloc[0]["change"] == "changed"

    def test_can_extend_from_head_false_on_non_head(self, tmp_path: Path) -> None:
        from jarl.experiments.graph import ExperimentGraph
        from jarl.training.config import RLRunConfig
        from tests.helpers.minimal_navix_configs import minimal_ppo_run_config

        exp_dir = tmp_path / "exp"
        config = minimal_ppo_run_config().apply_overrides({"algorithm.total_timesteps": 64})
        graph = ExperimentGraph(exp_dir, base_config=config)
        root = graph.create_root(config, label="root")
        graph.save()
        graph = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
        graph.checkout(root.id)
        successor = graph.extend(label="step2")
        graph.save()
        graph = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
        assert can_extend_from_head(graph, successor.id) is True
        assert can_extend_from_head(graph, root.id) is False


class TestRunContinuation:
    def test_extend_from_branch_head(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        config = minimal_ppo_run_config().apply_overrides({"algorithm.total_timesteps": 64})
        graph = ExperimentGraph(exp_dir, base_config=config)
        root = graph.create_root(config, label="root", prepare=True)
        graph.save()

        result = run_continuation(
            exp_dir,
            request=ContinuationRequest(
                kind="extend",
                from_node=root.id,
                branch="main",
                label="step2",
                prepare=True,
                checkpoint_pick=CheckpointPick.NONE,
            ),
            checkpoint_ref=None,
        )

        reloaded = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
        assert reloaded.head("main").id == result.child_id
        assert reloaded.get_node(result.child_id).node_metadata.parent_id == root.id

    def test_extend_rejects_non_head_origin(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        config = minimal_ppo_run_config().apply_overrides({"algorithm.total_timesteps": 64})
        graph = ExperimentGraph(exp_dir, base_config=config)
        root = graph.create_root(config, label="root", prepare=True)
        graph.extend(label="step2", prepare=True)
        graph.save()

        with pytest.raises(ValueError, match="not the head"):
            run_continuation(
                exp_dir,
                request=ContinuationRequest(
                    kind="extend",
                    from_node=root.id,
                    branch="main",
                    label="step3",
                    prepare=True,
                    checkpoint_pick=CheckpointPick.NONE,
                ),
                checkpoint_ref=None,
            )
