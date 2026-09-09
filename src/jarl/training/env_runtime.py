"""Environment runtime compatibility helpers for checkpoint transfer."""

from __future__ import annotations

from jarl.training.config import EnvironmentConfig

__all__ = ["env_io_contract_unchanged"]


def env_io_contract_unchanged(parent: EnvironmentConfig, child: EnvironmentConfig) -> bool:
    """Return whether two environment configs share the same observation/action contract.

    Transfer groups are validated separately during graph fork/extend. Reward mix and
    scenario overlay change dynamics without changing I/O, so they are excluded.
    """
    return (
        parent.env_id == child.env_id
        and parent.seed == child.seed
        and parent.nr_envs == child.nr_envs
        and parent.max_episode_steps == child.max_episode_steps
    )
