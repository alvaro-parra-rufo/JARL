"""Optimizer rebinding after fork/extend checkpoint restore."""

from __future__ import annotations

import jax
from flax.training.train_state import TrainState

__all__ = [
    "read_optimizer_learning_rate",
    "rebind_optimizer_after_fork_restore",
]


def read_optimizer_learning_rate(train_state: TrainState) -> jax.Array:
    """Return the current learning rate from a PPO ``inject_hyperparams`` train state."""
    return train_state.opt_state[1].hyperparams["learning_rate"]


def rebind_optimizer_after_fork_restore(
    restored_state: TrainState,
    template_state: TrainState,
) -> TrainState:
    """Rebind fork-restored parameters to the child node's optimizer.

    Fork and extend restore parent ``params`` and parent ``opt_state``. The parent
    optimizer step counter makes the child's LR schedule start out of budget (LR 0
    or negative). This helper keeps checkpoint ``params`` and ``step`` but
    reinitializes ``opt_state`` from ``template_state.tx`` so the child schedule
    starts at step zero.

    Intra-node resume must not call this helper; resume should preserve optimizer
    state and the LR counter from the same node.
    """
    return template_state.replace(
        step=restored_state.step,
        params=restored_state.params,
        opt_state=template_state.tx.init(restored_state.params),
    )
