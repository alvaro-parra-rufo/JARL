"""Tests for greedy Navix video rollout horizon alignment."""

from __future__ import annotations

import jax.numpy as jnp

from jarl.agents.ppo.video.rollout import NavixGreedyVideoRollout


class TestNavixGreedyVideoRolloutHorizon:
    """Greedy rollout should honor the training episode limit."""

    def test_max_episode_steps_configures_navix_env(self) -> None:
        rollout = NavixGreedyVideoRollout(
            env_id="Navix-Empty-5x5-v0",
            preprocess_observation=lambda observation: observation,
            policy_apply=lambda _params, observation: jnp.zeros((observation.shape[0], 1)),
            max_steps=12,
            max_episode_steps=12,
        )

        assert rollout.max_steps == 12
        assert int(rollout._env.max_steps) == 12
