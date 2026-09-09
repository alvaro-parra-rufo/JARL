"""Backend contracts for restoring policies used by inference."""

from __future__ import annotations

from typing import Protocol

import jax

from jarl.experiments.node import NodeWorkspace
from jarl.inference.policy import LoadedInferencePolicy
from jarl.training.config import RLRunConfig

__all__ = [
    "InferenceBackend",
    "InferenceEnvironment",
]


class InferenceEnvironment(Protocol):
    """Environment surface required while preparing an inference policy."""

    def preprocess_observation(self, observation: jax.Array) -> jax.Array:
        """Convert raw batched observations into policy inputs."""
        ...


class InferenceBackend(Protocol):
    """Callable that restores one algorithm-specific policy for inference."""

    def __call__(
        self,
        workspace: NodeWorkspace,
        config: RLRunConfig,
        checkpoint_step: int,
        env: InferenceEnvironment,
    ) -> LoadedInferencePolicy:
        """Load policy parameters and construct their reusable runtime."""
        ...
