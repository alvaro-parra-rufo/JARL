"""Regression tests for checkpoint train-state restore after Orbax deserialization."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jarl.experiments.checkpoint_state import restore_opt_state, restore_train_state, train_state_to_leaf
from tests.helpers.checkpoint_restore import orbax_like_opt_state, train_state_with_lr_schedule


class TestRestoreOptState:
    """Orbax must not leave optax optimizer state as plain dicts."""

    def test_restore_opt_state_rebuilds_inject_hyperparams_namedtuple(self) -> None:
        template = train_state_with_lr_schedule({"policy": jnp.array([1.0])})
        orbax_payload = orbax_like_opt_state(template.opt_state)

        rebuilt = restore_opt_state(template.opt_state, orbax_payload)

        assert not isinstance(rebuilt[1], dict)
        assert hasattr(rebuilt[1], "hyperparams")

    def test_restore_train_state_allows_optimizer_update_after_orbax_payload(self) -> None:
        source = train_state_with_lr_schedule({"policy": jnp.array([1.0, 2.0])}, step=3)
        template = train_state_with_lr_schedule({"policy": jnp.array([0.0, 0.0])})
        leaf = train_state_to_leaf(source)
        leaf_dict = leaf.to_dict()
        leaf_dict["opt_state"] = orbax_like_opt_state(source.opt_state)

        restored = restore_train_state(template, leaf_dict)
        grads = jax.tree.map(jnp.zeros_like, restored.params)

        restored.apply_gradients(grads=grads)

        assert hasattr(restored.opt_state[1], "hyperparams")
