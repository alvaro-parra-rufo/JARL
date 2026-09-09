"""Shared PPO full-JAX training primitives."""

from jarl.agents.ppo.core.batch import flatten_rollout_tensor
from jarl.agents.ppo.core.gae import compute_gae_advantages
from jarl.agents.ppo.core.hyperparameters import PPOHyperparameters
from jarl.agents.ppo.core.loss import PPOLossMetrics, make_ppo_loss_fn, mean_loss_metrics, ppo_loss_and_metrics
from jarl.agents.ppo.core.metrics import (
    compute_explained_variance,
    compute_policy_std_dev,
)
from jarl.agents.ppo.core.minibatch import (
    make_minibatch_indices,
    normalize_advantages,
)

__all__ = [
    "PPOHyperparameters",
    "PPOLossMetrics",
    "compute_explained_variance",
    "compute_gae_advantages",
    "compute_policy_std_dev",
    "flatten_rollout_tensor",
    "make_minibatch_indices",
    "make_ppo_loss_fn",
    "mean_loss_metrics",
    "normalize_advantages",
    "ppo_loss_and_metrics",
]
