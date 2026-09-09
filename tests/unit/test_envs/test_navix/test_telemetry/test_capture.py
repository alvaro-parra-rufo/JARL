"""Tests for compiled Navix telemetry capture."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import navix as nx
import numpy as np
import pytest

from jarl.agents.ppo.video.recurrent_rollout import NavixGreedyRecurrentVideoRollout
from jarl.agents.ppo.video.rollout import NavixGreedyVideoRollout
from jarl.envs.navix.aliases import resolve_navix_registry_id
from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.envs.navix.scenario_rewards import ScenarioRewardSpec
from jarl.envs.navix.telemetry import (
    NAVIX_EVENT_SLOT_NAMES,
    NavixCaptureProfile,
    NavixTelemetryRollout,
    derive_events,
)
from jarl.inference.policy import InferencePolicyRuntime, PolicyState

_ENV_ID = "Navix-Empty-5x5-v0"

# Compiles Navix rollouts; `make test` runs slow markers serially.
pytestmark = pytest.mark.slow


@pytest.fixture()
def sequence_runtime() -> InferencePolicyRuntime:
    """Policy runtime whose params contain a deterministic action sequence."""

    def preprocess_observation(observation: jax.Array) -> jax.Array:
        return jnp.asarray(observation, dtype=jnp.float32).reshape((observation.shape[0], -1)) / 255.0

    def initial_state(batch_size: int) -> jax.Array:
        return jnp.zeros((batch_size,), dtype=jnp.int32)

    def greedy_step(
        params: object,
        _observation: jax.Array,
        state: PolicyState,
    ) -> tuple[jax.Array, PolicyState]:
        action_sequence = jnp.asarray(params, dtype=jnp.int32)
        indices = jnp.minimum(jnp.asarray(state, dtype=jnp.int32), action_sequence.shape[0] - 1)
        return action_sequence[indices], jnp.asarray(state, dtype=jnp.int32) + 1

    return InferencePolicyRuntime(
        preprocess_observation=preprocess_observation,
        initial_state=initial_state,
        greedy_step=greedy_step,
    )


class TestNavixTelemetryRollout:
    """The capturer aligns state and transition arrays without Python callbacks."""

    def test_analysis_profile_captures_batched_state_and_transition_arrays(
        self,
        sequence_runtime: InferencePolicyRuntime,
    ) -> None:
        max_steps = 3
        rollout = NavixTelemetryRollout(
            _ENV_ID,
            sequence_runtime,
            max_steps,
            NavixCaptureProfile.analysis(),
        )
        keys = jax.random.split(jax.random.PRNGKey(0), 2)

        trace = rollout.rollout(jnp.asarray([1, 1, 1]), keys)

        assert trace.full_symbolic is not None
        assert trace.full_symbolic.shape[:2] == (2, max_steps + 1)
        assert trace.first_person_symbolic is not None
        assert trace.policy_observations is not None
        assert trace.player_positions is not None
        assert trace.player_positions.shape == (2, max_steps + 1, 2)
        assert trace.actions.shape == (2, max_steps)
        assert trace.event_happened is not None
        assert trace.event_happened.shape == (2, max_steps, len(NAVIX_EVENT_SLOT_NAMES))
        assert trace.rgb_frames is None
        np.testing.assert_array_equal(trace.actions, np.ones((2, max_steps), dtype=np.int32))
        np.testing.assert_array_equal(trace.episode_lengths, np.full((2,), max_steps))
        assert trace.player_directions is not None
        direction_changes = (np.diff(trace.player_directions, axis=1) % 4).astype(np.int32)
        np.testing.assert_array_equal(direction_changes, np.ones((2, max_steps), dtype=np.int32))
        expected_timestep = nx.make(
            resolve_navix_registry_id(_ENV_ID),
            observation_fn=nx.observations.symbolic_first_person,
        ).reset(keys[0])
        np.testing.assert_array_equal(
            trace.full_symbolic[0, 0],
            nx.observations.symbolic(expected_timestep.state),
        )
        np.testing.assert_allclose(
            trace.policy_observations[0, 0],
            sequence_runtime.preprocess_observation(expected_timestep.observation[None])[0],
        )
        host_trace = trace.to_host()
        assert isinstance(host_trace.actions, np.ndarray)
        assert isinstance(host_trace.full_symbolic, np.ndarray)

    def test_custom_mix_and_overlay_use_effective_reward(
        self,
        sequence_runtime: InferencePolicyRuntime,
    ) -> None:
        spec = ScenarioRewardSpec(
            id="test.overlay",
            version=1,
            scale=0.5,
            compatible_env_ids=(_ENV_ID,),
            position=(1, 2),
        )
        native = NavixTelemetryRollout(
            _ENV_ID,
            sequence_runtime,
            1,
            NavixCaptureProfile.analysis(),
        )
        overlay = NavixTelemetryRollout(
            _ENV_ID,
            sequence_runtime,
            1,
            NavixCaptureProfile.analysis(),
            reward=RewardWeightsConfig(),
            scenario_spec=spec,
        )
        keys = jax.random.split(jax.random.PRNGKey(0), 1)

        native_trace = native.rollout(jnp.asarray([2]), keys)
        overlay_trace = overlay.rollout(jnp.asarray([2]), keys)

        np.testing.assert_allclose(native_trace.rewards, np.zeros((1, 1), dtype=np.float32))
        np.testing.assert_allclose(overlay_trace.rewards, np.full((1, 1), 0.5, dtype=np.float32))

    def test_video_profile_omits_analysis_arrays(
        self,
        sequence_runtime: InferencePolicyRuntime,
    ) -> None:
        rollout = NavixTelemetryRollout(
            _ENV_ID,
            sequence_runtime,
            2,
            NavixCaptureProfile.video("first_person"),
        )

        trace = rollout.rollout(jnp.asarray([1, 1]), jax.random.split(jax.random.PRNGKey(1), 1))

        assert trace.rgb_frames is not None
        assert trace.rgb_frames.shape[1] == 3
        assert trace.full_symbolic is None
        assert trace.first_person_symbolic is None
        assert trace.policy_observations is None
        assert trace.player_positions is None
        assert trace.event_happened is None

    def test_scan_stops_environment_after_early_termination(
        self,
        sequence_runtime: InferencePolicyRuntime,
    ) -> None:
        key = jax.random.PRNGKey(2)
        action_sequence = _actions_to_goal(key)
        padded_actions = (*action_sequence, 1, 1, 1)
        rollout = NavixTelemetryRollout(
            _ENV_ID,
            sequence_runtime,
            len(padded_actions),
            NavixCaptureProfile.analysis(),
        )

        trace = rollout.rollout(jnp.asarray(padded_actions), key[None])
        length = int(trace.episode_lengths[0])

        assert length == len(action_sequence)
        assert bool(trace.episode_done[0]) is True
        np.testing.assert_array_equal(
            trace.active_mask[0],
            np.asarray([True] * length + [False] * 3),
        )
        np.testing.assert_array_equal(trace.actions[0, length:], -np.ones((3,), dtype=np.int32))
        assert trace.player_positions is not None
        final_position = trace.player_positions[0, length]
        np.testing.assert_array_equal(
            trace.player_positions[0, length + 1 :],
            np.repeat(final_position[None], 3, axis=0),
        )

    def test_first_transition_done_keeps_only_one_active_step(
        self,
        sequence_runtime: InferencePolicyRuntime,
    ) -> None:
        key = jax.random.PRNGKey(6)
        rollout = NavixTelemetryRollout(
            _ENV_ID,
            sequence_runtime,
            3,
            NavixCaptureProfile.analysis(),
            max_episode_steps=1,
        )

        trace = rollout.rollout(jnp.asarray([1, 1, 1]), key[None])

        assert int(trace.episode_lengths[0]) == 1
        np.testing.assert_array_equal(trace.active_mask[0], np.asarray([True, False, False]))
        np.testing.assert_array_equal(trace.actions[0], np.asarray([1, -1, -1]))

    def test_wall_hit_pulse_is_counted_once(
        self,
        sequence_runtime: InferencePolicyRuntime,
    ) -> None:
        key = jax.random.PRNGKey(4)
        action_sequence, hit_timestep = _actions_hit_north_wall_then_rotate(key)
        rollout = NavixTelemetryRollout(
            _ENV_ID,
            sequence_runtime,
            len(action_sequence),
            NavixCaptureProfile.analysis(),
        )

        trace = rollout.rollout(jnp.asarray(action_sequence), key[None])
        events = derive_events(trace, action_names=rollout.action_names)

        assert trace.event_happened is not None
        wall_hit_slot = NAVIX_EVENT_SLOT_NAMES.index("wall_hit")
        assert bool(trace.event_happened[0, hit_timestep, wall_hit_slot]) is True
        assert bool(trace.event_happened[0, hit_timestep + 1, wall_hit_slot]) is False
        assert sum(event.kind == "wall_hit" and event.source == "navix" for event in events) == 1
        assert sum(event.kind == "movement_blocked" for event in events) == 1

    def test_rejects_unbatched_keys(self, sequence_runtime: InferencePolicyRuntime) -> None:
        rollout = NavixTelemetryRollout(
            _ENV_ID,
            sequence_runtime,
            2,
            NavixCaptureProfile.analysis(),
        )

        with pytest.raises(ValueError, match="keys must have shape"):
            rollout.rollout(jnp.asarray([1, 1]), jax.random.PRNGKey(0))

    def test_same_seed_is_deterministic_and_random_seeds_can_change_state(
        self,
        sequence_runtime: InferencePolicyRuntime,
    ) -> None:
        rollout = NavixTelemetryRollout(
            "Navix-Empty-Random-5x5-v0",
            sequence_runtime,
            1,
            NavixCaptureProfile.analysis(),
        )
        keys = jax.random.split(jax.random.PRNGKey(12), 4)

        first = rollout.rollout(jnp.asarray([1]), keys)
        repeated = rollout.rollout(jnp.asarray([1]), keys)

        assert first.full_symbolic is not None
        assert repeated.full_symbolic is not None
        np.testing.assert_array_equal(first.full_symbolic, repeated.full_symbolic)
        unique_initial_states = np.unique(np.asarray(first.full_symbolic[:, 0]), axis=0)
        assert len(unique_initial_states) > 1


class TestVideoRolloutParity:
    """Reduced telemetry profiles preserve existing video rollout outputs."""

    def test_feedforward_frames_and_outcome_match_existing_rollout(self) -> None:
        runtime = _feedforward_logits_runtime()
        params = jnp.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        keys = jax.random.split(jax.random.PRNGKey(8), 2)
        telemetry = NavixTelemetryRollout(
            _ENV_ID,
            runtime,
            4,
            NavixCaptureProfile.video(),
            max_episode_steps=4,
        )
        existing = NavixGreedyVideoRollout(
            env_id=_ENV_ID,
            preprocess_observation=runtime.preprocess_observation,
            policy_apply=_logits_apply,
            max_steps=4,
            max_episode_steps=4,
        )

        trace = telemetry.rollout(params, keys)
        video = existing.rollout(params, keys)

        assert trace.rgb_frames is not None
        np.testing.assert_array_equal(trace.rgb_frames, video.frames)
        np.testing.assert_allclose(trace.episode_returns, video.episode_returns)
        np.testing.assert_array_equal(trace.episode_lengths, video.episode_lengths)
        np.testing.assert_array_equal(trace.episode_done, video.episode_done)

    def test_recurrent_frames_and_outcome_match_existing_rollout(self) -> None:
        runtime = _recurrent_logits_runtime()
        params = jnp.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        keys = jax.random.split(jax.random.PRNGKey(9), 2)
        telemetry = NavixTelemetryRollout(
            _ENV_ID,
            runtime,
            4,
            NavixCaptureProfile.video(),
            max_episode_steps=4,
        )
        existing = NavixGreedyRecurrentVideoRollout(
            env_id=_ENV_ID,
            preprocess_observation=runtime.preprocess_observation,
            policy_apply_one_step=_recurrent_logits_apply,
            initialize_carry=runtime.initial_state,
            max_steps=4,
            max_episode_steps=4,
        )

        trace = telemetry.rollout(params, keys)
        video = existing.rollout(params, keys)

        assert trace.rgb_frames is not None
        np.testing.assert_array_equal(trace.rgb_frames, video.frames)
        np.testing.assert_allclose(trace.episode_returns, video.episode_returns)
        np.testing.assert_array_equal(trace.episode_lengths, video.episode_lengths)
        np.testing.assert_array_equal(trace.episode_done, video.episode_done)


def _actions_to_goal(key: jax.Array) -> tuple[int, ...]:
    env = nx.make(
        resolve_navix_registry_id(_ENV_ID),
        max_steps=32,
        observation_fn=nx.observations.symbolic_first_person,
    )
    state = env.reset(key).state
    player = state.get_player()
    goal = state.get_goals()
    start = np.asarray(player.position, dtype=np.int32)
    target = np.asarray(goal.position[0], dtype=np.int32)
    direction = int(player.direction)
    actions: list[int] = []

    row_delta = int(target[0] - start[0])
    if row_delta:
        direction = _append_move(actions, direction, 1 if row_delta > 0 else 3, abs(row_delta))
    column_delta = int(target[1] - start[1])
    if column_delta:
        _append_move(actions, direction, 0 if column_delta > 0 else 2, abs(column_delta))
    return tuple(actions)


def _actions_hit_north_wall_then_rotate(key: jax.Array) -> tuple[tuple[int, ...], int]:
    env = nx.make(
        resolve_navix_registry_id(_ENV_ID),
        max_steps=32,
        observation_fn=nx.observations.symbolic_first_person,
    )
    player = env.reset(key).state.get_player()
    rotations = (3 - int(player.direction)) % 4
    forwards = int(player.position[0])
    actions = (*([1] * rotations), *([2] * forwards), 1)
    return tuple(actions), rotations + forwards - 1


def _append_move(
    actions: list[int],
    current_direction: int,
    target_direction: int,
    distance: int,
) -> int:
    rotations = (target_direction - current_direction) % 4
    actions.extend([1] * rotations)
    actions.extend([2] * distance)
    return target_direction


def _preprocess_observation(observation: jax.Array) -> jax.Array:
    return jnp.asarray(observation, dtype=jnp.float32).reshape((observation.shape[0], -1)) / 255.0


def _logits_apply(params: object, observation: jax.Array) -> jax.Array:
    logits = jnp.asarray(params, dtype=jnp.float32)
    return jnp.broadcast_to(logits, (observation.shape[0], logits.shape[0]))


def _recurrent_logits_apply(
    params: object,
    observation: jax.Array,
    carry: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return _logits_apply(params, observation), carry + 1


def _feedforward_logits_runtime() -> InferencePolicyRuntime:
    def greedy_step(
        params: object,
        observation: jax.Array,
        state: PolicyState,
    ) -> tuple[jax.Array, PolicyState]:
        return jnp.argmax(_logits_apply(params, observation), axis=-1), state

    return InferencePolicyRuntime(
        preprocess_observation=_preprocess_observation,
        initial_state=lambda _batch_size: (),
        greedy_step=greedy_step,
    )


def _recurrent_logits_runtime() -> InferencePolicyRuntime:
    def initial_state(batch_size: int) -> jax.Array:
        return jnp.zeros((batch_size, 1), dtype=jnp.float32)

    def greedy_step(
        params: object,
        observation: jax.Array,
        state: PolicyState,
    ) -> tuple[jax.Array, PolicyState]:
        logits, next_state = _recurrent_logits_apply(
            params,
            observation,
            jnp.asarray(state),
        )
        return jnp.argmax(logits, axis=-1), next_state

    return InferencePolicyRuntime(
        preprocess_observation=_preprocess_observation,
        initial_state=initial_state,
        greedy_step=greedy_step,
    )
