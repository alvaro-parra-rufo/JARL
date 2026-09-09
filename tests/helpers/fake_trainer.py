"""Fake trainer for runner smoke tests."""

from __future__ import annotations

from jarl.experiments.io.artifacts import ArtifactRecord
from jarl.experiments.node import NodeWorkspace
from jarl.training.config import RLRunConfig
from jarl.training.schedule import TrainingSchedule
from jarl.training.trainer import EnvFactory

__all__ = ["fake_trainer"]


def fake_trainer(
    workspace: NodeWorkspace,
    config: RLRunConfig,
    schedule: TrainingSchedule,
    *,
    env_factory: EnvFactory | None = None,
) -> None:
    """Write a smoke metric and register a dummy artifact.

    Args:
        workspace: Active node workspace inside the training context manager.
        config: Resolved RL run config with schedule fields populated.
        schedule: Derived training schedule counters.
        env_factory: Optional environment factory (ignored by the fake trainer).
    """
    del config, env_factory
    workspace.log_scalar(0, "runner_smoke", float(schedule.actual_rollout_updates))
    dummy_path = workspace.path / "dummy.txt"
    dummy_path.write_text("ok", encoding="utf-8")
    workspace.register_artifact(
        ArtifactRecord(
            name="dummy",
            kind="smoke",
            relative_path="dummy.txt",
            step=0,
        )
    )
