"""Shared helpers for checkpoint restore regression tests."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

__all__ = [
    "orbax_like_opt_state",
    "standard_restore_args_from_call",
    "train_state_with_lr_schedule",
]


def train_state_with_lr_schedule(params: dict[str, jnp.ndarray], *, step: int = 0) -> TrainState:
    """Build a train state with inject_hyperparams Adam for opt_state restore tests."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.inject_hyperparams(optax.adam)(learning_rate=1e-3),
    )
    state = TrainState.create(apply_fn=lambda *_args, **_kwargs: None, params=params, tx=optimizer)
    grads = jax.tree.map(jnp.zeros_like, params)
    state = state.apply_gradients(grads=grads)
    return state.replace(step=jnp.asarray(step, dtype=jnp.int32))


def orbax_like_opt_state(opt_state: object) -> object:
    """Simulate Orbax deserialization of optax state into dicts and lists."""
    return _to_orbax(opt_state)


def standard_restore_args_from_call(call: Any) -> Any:
    """Return the ``StandardRestore`` args object passed to an Orbax ``restore`` call."""
    if call is None:
        msg = "Expected an Orbax restore call, got None."
        raise AssertionError(msg)
    if call.kwargs:
        return call.kwargs["args"]
    return call.args[1]


def _to_orbax(value: object) -> object:
    fields = getattr(type(value), "_fields", None)
    if fields is not None:
        return {field: _to_orbax(getattr(value, field)) for field in fields}
    if isinstance(value, dict):
        return {key: _to_orbax(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_orbax(item) for item in value]
    return value
