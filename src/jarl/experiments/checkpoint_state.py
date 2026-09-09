"""Typed training checkpoint payloads persisted through Orbax."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax
from flax.training.train_state import TrainState

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "TrainStateLeaf",
    "TrainingCheckpoint",
    "deserialize_rng_key",
    "restore_opt_state",
    "restore_train_state",
    "serialize_rng_key",
    "train_state_to_leaf",
]

CHECKPOINT_FORMAT_VERSION = 1
"""Current jarl training checkpoint envelope version."""

_PRNG_KEY_PARTS = 2


@runtime_checkable
class TrainingCheckpoint(Protocol):
    """Protocol for algorithm-specific Orbax checkpoint payloads."""

    version: int
    kind: str
    algorithm_name: str
    global_step: int
    optimizer_updates: int
    rng_key: tuple[int, ...]

    def to_pytree(self) -> dict[str, Any]:
        """Serialize the checkpoint into an Orbax-compatible dict pytree."""

    @classmethod
    def from_pytree(cls, tree: dict[str, Any]) -> TrainingCheckpoint:
        """Reconstruct a checkpoint from an Orbax-restored dict pytree."""


@dataclass(frozen=True, slots=True)
class TrainStateLeaf:
    """Serializable subset of a Flax ``TrainState``."""

    step: int
    params: Any
    opt_state: Any

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/Orbax-friendly mapping."""
        return {"step": self.step, "params": self.params, "opt_state": self.opt_state}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainStateLeaf:
        """Build a leaf from a restored checkpoint mapping."""
        return cls(step=int(payload["step"]), params=payload["params"], opt_state=payload["opt_state"])


def train_state_to_leaf(state: TrainState) -> TrainStateLeaf:
    """Extract the serializable subset of a ``TrainState``."""
    return TrainStateLeaf(step=int(state.step), params=state.params, opt_state=state.opt_state)


def restore_opt_state(template: Any, restored: Any) -> Any:  # noqa: PLR0911
    """Rebuild optax optimizer state from Orbax-deserialized leaves.

    Orbax restore may return nested dicts and lists instead of optax ``NamedTuple``
    states. ``template`` supplies the expected pytree structure and types.
    """
    if restored is None:
        return template

    fields = getattr(type(template), "_fields", None)
    if fields is not None:
        if isinstance(restored, dict):
            return type(template)(
                **{field: restore_opt_state(getattr(template, field), restored[field]) for field in fields}
            )
        return restored

    if isinstance(template, dict):
        if isinstance(restored, dict):
            return {key: restore_opt_state(template[key], restored[key]) for key in template}
        return restored

    if isinstance(template, (tuple, list)):
        restored_seq = restored
        if isinstance(restored, dict):
            if restored and all(isinstance(key, str) and key.isdigit() for key in restored):
                restored_seq = tuple(restored[str(index)] for index in range(len(template)))
            else:
                restored_seq = tuple(restored.values())
        elif isinstance(restored, list):
            restored_seq = tuple(restored)
        if not isinstance(restored_seq, tuple):
            restored_seq = tuple(restored_seq)
        return type(template)(
            restore_opt_state(template_leaf, restored_leaf)
            for template_leaf, restored_leaf in zip(template, restored_seq, strict=True)
        )

    return restored


def restore_train_state(template: TrainState, leaf: TrainStateLeaf | dict[str, Any]) -> TrainState:
    """Rebuild a ``TrainState`` from a template and a serialized leaf."""
    resolved = leaf if isinstance(leaf, TrainStateLeaf) else TrainStateLeaf.from_dict(leaf)
    step_value = template.step
    step_dtype = step_value.dtype if hasattr(step_value, "dtype") else jax.numpy.int32
    params = jax.tree.map(
        lambda template_leaf, restored_leaf: (
            jax.device_put(restored_leaf, template_leaf.sharding)
            if isinstance(template_leaf, jax.Array)
            else restored_leaf
        ),
        template.params,
        resolved.params,
    )
    return template.replace(
        step=jax.numpy.asarray(resolved.step, dtype=step_dtype),
        params=params,
        opt_state=restore_opt_state(template.opt_state, resolved.opt_state),
    )


def serialize_rng_key(key: jax.Array) -> tuple[int, ...]:
    """Serialize a PRNG key into host integers."""
    return tuple(int(value) for value in jax.device_get(jax.random.key_data(key)).reshape(-1))


def deserialize_rng_key(data: tuple[int, ...] | list[int]) -> jax.Array:
    """Deserialize host integers back into a scalar PRNG key."""
    values = tuple(int(value) for value in data)
    if len(values) != _PRNG_KEY_PARTS:
        msg = f"PRNG key payload must contain exactly two integers, got {len(values)}."
        raise ValueError(msg)
    key_data = jax.numpy.asarray(values, dtype=jax.numpy.uint32)
    return jax.random.wrap_key_data(key_data)
