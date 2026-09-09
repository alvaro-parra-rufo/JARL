"""Restore PPO training state from typed parent checkpoints."""

from __future__ import annotations

import json

import jax
from flax.training.train_state import TrainState

from jarl.agents.ppo.checkpoint_state import (
    PPOCheckpoint,
    PPOGrUCheckpoint,
    checkpoint_from_pytree,
    checkpoint_kind_for_algorithm,
)
from jarl.experiments.io.layout import CONFIG_FILENAME
from jarl.experiments.node import NodeWorkspace
from jarl.training.config import EnvironmentConfig, RLRunConfig

__all__ = [
    "load_parent_environment_config",
    "restore_fork_parent_checkpoint",
]


def load_parent_environment_config(workspace: NodeWorkspace) -> EnvironmentConfig | None:
    """Load the parent node's resolved environment config when available."""
    parent_id = workspace.node_metadata.parent_id
    if parent_id is None:
        return None
    parent_config_path = workspace.layout.root.parent / parent_id / CONFIG_FILENAME
    if not parent_config_path.is_file():
        return None
    payload = json.loads(parent_config_path.read_text(encoding="utf-8"))
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        return None
    return EnvironmentConfig.model_validate(environment)


def restore_fork_parent_checkpoint(
    *,
    workspace: NodeWorkspace,
    config: RLRunConfig,
    policy_template: TrainState,
    critic_template: TrainState,
    zero_carry: jax.Array | None = None,
) -> tuple[jax.Array, TrainState, TrainState, jax.Array | None, jax.Array | None] | None:
    """Restore training state from a parent checkpoint referenced by the workspace.

    Returns ``None`` when the node is not forked from a parent checkpoint. Raises
    when a parent checkpoint is required but missing or incompatible.
    """
    if workspace.parent_checkpoint_step is None:
        return None

    tree = workspace.load_parent_checkpoint()
    if not isinstance(tree, dict):
        msg = "Parent checkpoint payload must be a mapping."
        raise TypeError(msg)

    checkpoint = checkpoint_from_pytree(tree)
    expected_kind = checkpoint_kind_for_algorithm(config.algorithm.name)
    if checkpoint.kind != expected_kind:
        msg = (
            f"Parent checkpoint kind {checkpoint.kind!r} does not match "
            f"algorithm {config.algorithm.name!r} ({expected_kind!r})."
        )
        raise ValueError(msg)

    if isinstance(checkpoint, PPOGrUCheckpoint):
        if zero_carry is None:
            msg = "PPO-GRU checkpoint restore requires a zero carry template."
            raise ValueError(msg)
        return checkpoint.restore_train_states(
            policy_template=policy_template,
            critic_template=critic_template,
            zero_carry=zero_carry,
        )

    if isinstance(checkpoint, PPOCheckpoint):
        key, policy_state, critic_state = checkpoint.restore_train_states(
            policy_template=policy_template,
            critic_template=critic_template,
        )
        return key, policy_state, critic_state, None, None

    msg = f"Unsupported checkpoint type: {type(checkpoint)!r}"
    raise TypeError(msg)
