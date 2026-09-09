"""Shared trainer restore logic for fork transfer and intra-node resume."""

from __future__ import annotations

from dataclasses import dataclass

import jax
from flax.training.train_state import TrainState

from jarl.agents.ppo.checkpoint_restore import restore_fork_parent_checkpoint
from jarl.agents.ppo.checkpoint_state import (
    PPOCheckpoint,
    PPOGrUCheckpoint,
    checkpoint_from_pytree,
    checkpoint_kind_for_algorithm,
)
from jarl.agents.ppo.optimizer_restore import rebind_optimizer_after_fork_restore
from jarl.experiments.node import NodeWorkspace
from jarl.training.config import RLRunConfig
from jarl.training.resume_budget import apply_remaining_timestep_budget
from jarl.training.schedule import TrainingSchedule

__all__ = ["TrainingRestoreResult", "prepare_training_restore"]


@dataclass(frozen=True, slots=True)
class TrainingRestoreResult:
    """Training state and schedule produced by restore helpers.

    Recurrent trainers restore network and optimizer state. Episodic GRU carries
    are always reset because ``train()`` starts from ``env.reset()``.
    """

    key: jax.Array
    policy_state: TrainState
    critic_state: TrainState
    policy_carry: jax.Array | None
    critic_carry: jax.Array | None
    config: RLRunConfig
    schedule: TrainingSchedule


def prepare_training_restore(
    *,
    workspace: NodeWorkspace,
    config: RLRunConfig,
    schedule: TrainingSchedule,
    policy_template: TrainState,
    critic_template: TrainState,
    zero_carry: jax.Array | None = None,
    key: jax.Array,
) -> TrainingRestoreResult:
    """Restore trainer state from resume or fork checkpoints when applicable."""
    attempt = workspace.latest_execution_attempt()
    if attempt is not None and attempt.resume_of_attempt_id is not None:
        return _restore_resume(
            workspace=workspace,
            config=config,
            policy_template=policy_template,
            critic_template=critic_template,
            zero_carry=zero_carry,
            _key=key,
        )

    restored = restore_fork_parent_checkpoint(
        workspace=workspace,
        config=config,
        policy_template=policy_template,
        critic_template=critic_template,
        zero_carry=zero_carry,
    )
    if restored is not None:
        restored_key, policy_state, critic_state, policy_carry, critic_carry = restored
        policy_state = rebind_optimizer_after_fork_restore(policy_state, policy_template)
        critic_state = rebind_optimizer_after_fork_restore(critic_state, critic_template)
        return TrainingRestoreResult(
            key=restored_key,
            policy_state=policy_state,
            critic_state=critic_state,
            policy_carry=policy_carry,
            critic_carry=critic_carry,
            config=config,
            schedule=schedule,
        )

    return TrainingRestoreResult(
        key=key,
        policy_state=policy_template,
        critic_state=critic_template,
        policy_carry=None,
        critic_carry=None,
        config=config,
        schedule=schedule,
    )


def _restore_resume(
    *,
    workspace: NodeWorkspace,
    config: RLRunConfig,
    policy_template: TrainState,
    critic_template: TrainState,
    zero_carry: jax.Array | None,
    _key: jax.Array,
) -> TrainingRestoreResult:
    tree = workspace.load_resume_checkpoint()
    if not isinstance(tree, dict):
        msg = "Resume checkpoint payload must be a mapping."
        raise TypeError(msg)

    checkpoint = checkpoint_from_pytree(tree)
    expected_kind = checkpoint_kind_for_algorithm(config.algorithm.name)
    if checkpoint.kind != expected_kind:
        msg = (
            f"Resume checkpoint kind {checkpoint.kind!r} does not match "
            f"algorithm {config.algorithm.name!r} ({expected_kind!r})."
        )
        raise ValueError(msg)

    adjusted_config, adjusted_schedule = apply_remaining_timestep_budget(config, checkpoint.global_step)

    if isinstance(checkpoint, PPOGrUCheckpoint):
        if zero_carry is None:
            msg = "PPO-GRU resume restore requires a zero carry template."
            raise ValueError(msg)
        restored_key, policy_state, critic_state, policy_carry, critic_carry = checkpoint.restore_train_states(
            policy_template=policy_template,
            critic_template=critic_template,
            zero_carry=zero_carry,
        )
        return TrainingRestoreResult(
            key=restored_key,
            policy_state=policy_state,
            critic_state=critic_state,
            policy_carry=policy_carry,
            critic_carry=critic_carry,
            config=adjusted_config,
            schedule=adjusted_schedule,
        )

    if isinstance(checkpoint, PPOCheckpoint):
        restored_key, policy_state, critic_state = checkpoint.restore_train_states(
            policy_template=policy_template,
            critic_template=critic_template,
        )
        return TrainingRestoreResult(
            key=restored_key,
            policy_state=policy_state,
            critic_state=critic_state,
            policy_carry=None,
            critic_carry=None,
            config=adjusted_config,
            schedule=adjusted_schedule,
        )

    msg = f"Unsupported resume checkpoint type: {type(checkpoint)!r}"
    raise TypeError(msg)
