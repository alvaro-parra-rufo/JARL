"""PPO full-JAX agents for jarl."""

from jarl.agents.ppo.action import (
    ProcessedActionFn,
    continuous_log_prob,
    discrete_log_prob,
    make_processed_action_fn,
    sample_continuous_action,
    sample_discrete_action,
)
from jarl.agents.ppo.core import (
    PPOHyperparameters,
    PPOLossMetrics,
    compute_explained_variance,
    compute_gae_advantages,
    compute_policy_std_dev,
    flatten_rollout_tensor,
    make_minibatch_indices,
    make_ppo_loss_fn,
    normalize_advantages,
)
from jarl.agents.ppo.core.gru_minibatch import make_gru_minibatch_env_indices
from jarl.agents.ppo.core.loss import mean_loss_metrics, ppo_loss_and_metrics
from jarl.agents.ppo.gru_trainer import ppo_gru_full_jax_trainer
from jarl.agents.ppo.networks import (
    ContinuousGrUPolicy,
    ContinuousPolicy,
    Critic,
    DiscreteGrUPolicy,
    DiscretePolicy,
    GrUCritic,
    create_action_processor,
    create_critic_network,
    create_gru_critic_network,
    create_gru_policy_network,
    create_policy_network,
)
from jarl.agents.ppo.trainer import ppo_full_jax_trainer

__all__ = [
    "ContinuousGrUPolicy",
    "ContinuousPolicy",
    "Critic",
    "DiscreteGrUPolicy",
    "DiscretePolicy",
    "GrUCritic",
    "PPOHyperparameters",
    "PPOLossMetrics",
    "ProcessedActionFn",
    "compute_explained_variance",
    "compute_gae_advantages",
    "compute_policy_std_dev",
    "continuous_log_prob",
    "create_action_processor",
    "create_critic_network",
    "create_gru_critic_network",
    "create_gru_policy_network",
    "create_policy_network",
    "discrete_log_prob",
    "flatten_rollout_tensor",
    "make_gru_minibatch_env_indices",
    "make_minibatch_indices",
    "make_ppo_loss_fn",
    "make_processed_action_fn",
    "mean_loss_metrics",
    "normalize_advantages",
    "ppo_full_jax_trainer",
    "ppo_gru_full_jax_trainer",
    "ppo_loss_and_metrics",
    "sample_continuous_action",
    "sample_discrete_action",
]
