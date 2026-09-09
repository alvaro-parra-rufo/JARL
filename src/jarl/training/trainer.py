"""Trainer protocol for jarl RL training runs."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from jarl.experiments.node import NodeWorkspace
from jarl.training.config import RLRunConfig
from jarl.training.schedule import TrainingSchedule

__all__ = [
    "EnvFactory",
    "ScheduleBuilder",
    "Trainer",
]


class EnvFactory(Protocol):
    """Factory stub for future Navix environment construction."""

    def __call__(self, config: RLRunConfig) -> Any:
        """Build a training environment from the resolved run config."""
        ...


class ScheduleBuilder(Protocol):
    """Callable that applies derived schedule fields to a resolved config."""

    def __call__(self, config: RLRunConfig) -> RLRunConfig:
        """Return config with derived algorithm schedule fields populated."""
        ...


@runtime_checkable
class Trainer(Protocol):
    """Train on an active node workspace using a resolved config and schedule."""

    def __call__(
        self,
        workspace: NodeWorkspace,
        config: RLRunConfig,
        schedule: TrainingSchedule,
        *,
        env_factory: EnvFactory | None = None,
    ) -> None:
        """Run training for the active workspace context."""
        ...
