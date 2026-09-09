"""Greedy Navix rollouts for recurrent PPO policy video generation."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from jarl.agents.ppo.video.rollout import NavixVideoRolloutBatch
from jarl.envs.navix.telemetry import (
    NavixCaptureProfile,
    NavixTelemetryRollout,
    VideoViewMode,
)
from jarl.inference.policy import InferencePolicyRuntime, PolicyState

__all__ = [
    "NavixGreedyRecurrentVideoRollout",
]


class NavixGreedyRecurrentVideoRollout:
    """Adapt recurrent policy callbacks to the shared telemetry rollout."""

    def __init__(
        self,
        env_id: str,
        preprocess_observation: Callable[[jax.Array], jax.Array],
        policy_apply_one_step: Callable[..., tuple[jax.Array, jax.Array]],
        initialize_carry: Callable[[int], jax.Array],
        max_steps: int,
        view_mode: VideoViewMode = "full",
        max_episode_steps: int | None = None,
    ) -> None:
        """Create a compiled recurrent rollout helper for a fixed Navix task and horizon."""
        self.env_id = env_id
        self.max_steps = int(max_steps)
        self.view_mode = view_mode

        def greedy_step(
            params: object,
            observation: jax.Array,
            state: PolicyState,
        ) -> tuple[jax.Array, PolicyState]:
            logits, next_state = policy_apply_one_step(
                params,
                observation,
                jnp.asarray(state),
            )
            return jnp.argmax(logits, axis=-1).astype(jnp.int32), next_state

        runtime = InferencePolicyRuntime(
            preprocess_observation=preprocess_observation,
            initial_state=initialize_carry,
            greedy_step=greedy_step,
        )
        self._telemetry = NavixTelemetryRollout(
            env_id,
            runtime,
            self.max_steps,
            NavixCaptureProfile.video(view_mode),
            max_episode_steps=max_episode_steps,
        )
        self._env = self._telemetry.raw_env

    def rollout(self, policy_params: object, keys: jax.Array) -> NavixVideoRolloutBatch:
        """Return a batch of fixed-length RGB rollouts for recurrent ``policy_params``."""
        trace = self._telemetry.rollout(policy_params, keys)
        if trace.rgb_frames is None:
            msg = "Video capture profile did not produce RGB frames."
            raise RuntimeError(msg)
        return NavixVideoRolloutBatch(
            frames=trace.rgb_frames,
            episode_returns=trace.episode_returns,
            episode_lengths=trace.episode_lengths,
            episode_done=trace.episode_done,
        )
