"""Default environment factories for RL training runs."""

from __future__ import annotations

from jarl.envs.navix import NavixFullJITEnv, make_navix_full_jit_env
from jarl.envs.navix.scenario_rewards import resolve_scenario_reward_spec
from jarl.training.config import RLRunConfig

__all__ = [
    "navix_full_jit_env_factory",
]


def navix_full_jit_env_factory(config: RLRunConfig) -> NavixFullJITEnv:
    """Build a Navix full-JIT environment from a resolved ``RLRunConfig``.

    Args:
        config: Resolved run config whose ``environment`` block supplies Navix
            identifiers and observation settings.

    Returns:
        Configured ``NavixFullJITEnv`` ready for batched reset and step.
    """
    environment = config.environment
    scenario_spec = resolve_scenario_reward_spec(
        environment.env_id,
        environment.scenario_reward_id,
        environment.scenario_reward_version,
    )
    return make_navix_full_jit_env(
        env_id=environment.env_id,
        max_episode_steps=environment.max_episode_steps,
        reward=environment.reward,
        scenario_spec=scenario_spec,
    )
