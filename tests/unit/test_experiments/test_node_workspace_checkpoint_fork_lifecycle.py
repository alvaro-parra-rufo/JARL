"""Integration tests for checkpoint, model archive, and fork lifecycle."""

from __future__ import annotations

from pathlib import Path

from jarl.config import BaseConfig
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import CHECKPOINT_ALIAS_FINAL, CHECKPOINT_ALIAS_LATEST, CheckpointRef
from jarl.experiments.io.model_archive import MODEL_ALIAS_LATEST
from jarl.experiments.run_config import CheckpointConfig


class LifecycleConfig(BaseConfig):
    """Concrete config for checkpoint fork lifecycle tests."""

    learning_rate: float = 0.01


class TestCheckpointForkLifecycle:
    """End-to-end checkpoint, model archive, and fork workflow."""

    def test_checkpoint_model_fork_aliases(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "runs" / "checkpoint_fork"
        graph: ExperimentGraph[LifecycleConfig] = ExperimentGraph(exp_dir)
        root = graph.create_root(config=LifecycleConfig(), branch="main", label="root")
        policy = CheckpointConfig(save_interval_steps=5)

        with root:
            root.init_checkpoint_manager(max_to_keep=None, save_interval_steps=100)
            root.log_scalar(5, "loss", 0.4)
            root.save_checkpoint_if_due(5, {"step": 5}, policy=policy, total_steps=10)
            root.save_model_archive(
                "weights",
                5,
                policy={"policy": 1},
                critic={"critic": 2},
                alias=MODEL_ALIAS_LATEST,
            )

        assert root.resolve_checkpoint_alias(CHECKPOINT_ALIAS_LATEST) is not None
        assert root.resolve_checkpoint_alias(CHECKPOINT_ALIAS_FINAL) is not None
        assert root.resolve_model_alias(MODEL_ALIAS_LATEST) is not None

        graph.fork(
            "branch_b",
            from_node=root,
            from_checkpoint=CheckpointRef(node_id=root.id, checkpoint_step=5),
        )
        graph.save()

        restored = ExperimentGraph.from_directory(exp_dir, config_cls=LifecycleConfig)
        child = restored.head("branch_b")

        assert child.node_metadata.parent_checkpoint_step == 5
        assert child.load_checkpoint() == {"step": 5}
