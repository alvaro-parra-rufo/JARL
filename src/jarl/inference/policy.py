"""Algorithm-independent policy runtime contracts for inference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax

PolicyState = object
"""Algorithm-specific policy state carried between inference steps."""

PreprocessObservationFn = Callable[[jax.Array], jax.Array]
"""Callable that converts raw batched observations into policy inputs."""

InitialPolicyStateFn = Callable[[int], PolicyState]
"""Callable that creates policy state for a requested batch size."""

GreedyPolicyStepFn = Callable[
    [object, jax.Array, PolicyState],
    tuple[jax.Array, PolicyState],
]
"""Callable that selects environment actions and advances policy state."""

__all__ = [
    "GreedyPolicyStepFn",
    "InferencePolicyRuntime",
    "InitialPolicyStateFn",
    "LoadedInferencePolicy",
    "PolicyState",
    "PreprocessObservationFn",
]


@dataclass(frozen=True, slots=True)
class InferencePolicyRuntime:
    """Stateless policy execution functions shared by inference consumers."""

    preprocess_observation: PreprocessObservationFn
    initial_state: InitialPolicyStateFn
    greedy_step: GreedyPolicyStepFn


@dataclass(frozen=True, slots=True)
class LoadedInferencePolicy:
    """Restored policy parameters, runtime functions, and checkpoint identity."""

    runtime: InferencePolicyRuntime
    params: object
    source_node_id: str
    checkpoint_step: int
    checkpoint_kind: str
    checkpoint_version: int
    algorithm_name: str
    global_step: int
    optimizer_updates: int
