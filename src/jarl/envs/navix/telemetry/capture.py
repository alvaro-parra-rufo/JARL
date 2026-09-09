"""Compiled, observational Navix rollout capture."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import navix as nx

from jarl.envs.navix.aliases import resolve_navix_registry_id
from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.envs.navix.scenario_rewards import ScenarioRewardSpec, effective_reward_fn
from jarl.envs.navix.telemetry.contracts import (
    NAVIX_EVENT_SLOT_SPECS,
    NAVIX_EVENT_TYPE_CODES,
    NavixCaptureProfile,
    NavixTraceArrays,
)
from jarl.inference.policy import InferencePolicyRuntime, PolicyState

__all__ = ["NavixTelemetryRollout"]

_KEY_ARRAY_RANK = 2
"""Expected rank for a batch of JAX PRNG keys."""

_ABSENT_POSITION = jnp.asarray([-1, -1], dtype=jnp.int32)
"""Sentinel ``(row, column)`` when an event slot did not fire."""

_ABSENT_SCALAR = jnp.asarray(-1, dtype=jnp.int32)
"""Sentinel colour or event-type code when an event slot did not fire."""


class _StateProjection(NamedTuple):
    full_symbolic: jax.Array | None
    first_person_symbolic: jax.Array | None
    policy_observation: jax.Array | None
    player_position: jax.Array | None
    player_direction: jax.Array | None
    player_pocket: jax.Array | None
    rgb_frame: jax.Array | None


class _TransitionProjection(NamedTuple):
    state: _StateProjection
    action: jax.Array
    reward: jax.Array
    step_type: jax.Array
    done: jax.Array
    active: jax.Array
    event_happened: jax.Array | None
    event_positions: jax.Array | None
    event_colours: jax.Array | None
    event_types: jax.Array | None


_RolloutCarry = tuple[nx.Timestep, PolicyState, jax.Array, jax.Array, jax.Array]


class NavixTelemetryRollout:
    """Run greedy policies while projecting selected Navix evidence arrays."""

    def __init__(
        self,
        env_id: str,
        runtime: InferencePolicyRuntime,
        max_steps: int,
        profile: NavixCaptureProfile,
        *,
        max_episode_steps: int | None = None,
        reward: RewardWeightsConfig | None = None,
        scenario_spec: ScenarioRewardSpec | None = None,
    ) -> None:
        """Create a compiled rollout for a fixed environment, horizon, and profile."""
        if max_steps <= 0:
            msg = f"max_steps must be > 0, got {max_steps}."
            raise ValueError(msg)
        if max_episode_steps is not None and max_episode_steps <= 0:
            msg = f"max_episode_steps must be > 0, got {max_episode_steps}."
            raise ValueError(msg)
        registry_env_id = resolve_navix_registry_id(env_id)
        self.env_id = env_id
        self.max_steps = int(max_steps)
        self.profile = profile
        self._runtime = runtime
        make_kwargs: dict[str, object] = {"observation_fn": nx.observations.symbolic_first_person}
        if max_episode_steps is not None:
            make_kwargs["max_steps"] = int(max_episode_steps)
        if reward is not None:
            make_kwargs["reward_fn"] = effective_reward_fn(reward, scenario_spec)
        self._env = nx.make(registry_env_id, **make_kwargs)
        self._action_names = tuple(action.__name__ for action in self._env.action_set)
        self._rollout_batch = jax.jit(jax.vmap(self._rollout_episode, in_axes=(None, 0)))

    @property
    def action_names(self) -> tuple[str, ...]:
        """Return action names in the environment's discrete index order."""
        return self._action_names

    @property
    def raw_env(self) -> nx.Environment:
        """Return the underlying raw Navix environment."""
        return self._env

    def rollout(self, params: object, keys: jax.Array) -> NavixTraceArrays:
        """Return fixed-shape batched traces for ``params`` and reset ``keys``."""
        if keys.ndim != _KEY_ARRAY_RANK:
            msg = f"keys must have shape (episodes, key_parts), got {keys.shape}."
            raise ValueError(msg)
        return self._rollout_batch(params, keys)

    def _rollout_episode(self, params: object, key: jax.Array) -> NavixTraceArrays:
        """Capture one fixed-horizon episode before outer vectorization."""
        timestep = self._env.reset(key)
        policy_state = self._runtime.initial_state(1)
        initial_state = self._project_state(timestep)
        mission_present, mission_position, mission_colour, mission_event_type = _project_mission(timestep.state)

        def step(
            carry: _RolloutCarry,
            _unused: None,
        ) -> tuple[_RolloutCarry, _TransitionProjection]:
            current_timestep, current_policy_state, done, episode_return, episode_length = carry
            processed_observation = self._runtime.preprocess_observation(current_timestep.observation[None])
            action_batch, candidate_policy_state = self._runtime.greedy_step(
                params,
                processed_observation,
                current_policy_state,
            )
            action = jnp.asarray(action_batch[0], dtype=jnp.int32)
            active = jnp.logical_not(done)

            next_timestep = jax.lax.cond(
                done,
                lambda inactive_timestep: inactive_timestep,
                lambda active_timestep: self._env.step(active_timestep, action),
                current_timestep,
            )
            next_done = done | next_timestep.is_done()
            next_policy_state = jax.tree.map(
                lambda candidate, current: jnp.where(active, candidate, current),
                candidate_policy_state,
                current_policy_state,
            )
            reward = jnp.where(active, next_timestep.reward, jnp.asarray(0.0, dtype=jnp.float32))
            next_return = episode_return + reward
            next_length = episode_length + jnp.where(active, 1, 0)
            projection = self._transition_projection(
                next_timestep,
                action=action,
                reward=reward,
                done=next_done,
                active=active,
            )
            next_carry = (
                next_timestep,
                next_policy_state,
                next_done,
                next_return,
                next_length,
            )
            return next_carry, projection

        (_, _, episode_done, episode_return, episode_length), transitions = jax.lax.scan(
            step,
            (
                timestep,
                policy_state,
                jnp.asarray(False),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
            None,
            length=self.max_steps,
        )
        return NavixTraceArrays(
            full_symbolic=_prepend_optional(initial_state.full_symbolic, transitions.state.full_symbolic),
            first_person_symbolic=_prepend_optional(
                initial_state.first_person_symbolic,
                transitions.state.first_person_symbolic,
            ),
            policy_observations=_prepend_optional(
                initial_state.policy_observation,
                transitions.state.policy_observation,
            ),
            player_positions=_prepend_optional(
                initial_state.player_position,
                transitions.state.player_position,
            ),
            player_directions=_prepend_optional(
                initial_state.player_direction,
                transitions.state.player_direction,
            ),
            player_pockets=_prepend_optional(
                initial_state.player_pocket,
                transitions.state.player_pocket,
            ),
            rgb_frames=_prepend_optional(initial_state.rgb_frame, transitions.state.rgb_frame),
            actions=transitions.action,
            rewards=transitions.reward,
            step_types=transitions.step_type,
            dones=transitions.done,
            active_mask=transitions.active,
            event_happened=transitions.event_happened,
            event_positions=transitions.event_positions,
            event_colours=transitions.event_colours,
            event_types=transitions.event_types,
            mission_present=mission_present,
            mission_position=mission_position,
            mission_colour=mission_colour,
            mission_event_type=mission_event_type,
            episode_returns=episode_return,
            episode_lengths=episode_length,
            episode_done=episode_done,
        )

    def _project_state(self, timestep: nx.Timestep) -> _StateProjection:
        """Project selected evidence from a Navix timestep."""
        state = timestep.state
        if self.profile.capture_symbolic:
            full_symbolic = nx.observations.symbolic(state)
            first_person_symbolic = nx.observations.symbolic_first_person(state)
        else:
            full_symbolic = None
            first_person_symbolic = None
        if self.profile.capture_policy_observation:
            policy_observation = self._runtime.preprocess_observation(timestep.observation[None])[0]
        else:
            policy_observation = None
        if self.profile.capture_player:
            player = state.get_player()
            player_position = jnp.asarray(player.position, dtype=jnp.int32)
            player_direction = jnp.asarray(player.direction, dtype=jnp.int32)
            player_pocket = jnp.asarray(player.pocket, dtype=jnp.int32)
        else:
            player_position = None
            player_direction = None
            player_pocket = None
        rgb_frame = self._render_rgb(state)
        return _StateProjection(
            full_symbolic=full_symbolic,
            first_person_symbolic=first_person_symbolic,
            policy_observation=policy_observation,
            player_position=player_position,
            player_direction=player_direction,
            player_pocket=player_pocket,
            rgb_frame=rgb_frame,
        )

    def _transition_projection(
        self,
        timestep: nx.Timestep,
        *,
        action: jax.Array,
        reward: jax.Array,
        done: jax.Array,
        active: jax.Array,
    ) -> _TransitionProjection:
        """Project one post-action state and its transition values."""
        if self.profile.capture_events:
            event_happened, event_positions, event_colours, event_types = _project_events(timestep.state.events)
            event_happened = jnp.where(active, event_happened, jnp.zeros_like(event_happened))
            event_positions = jnp.where(active, event_positions, -jnp.ones_like(event_positions))
            event_colours = jnp.where(active, event_colours, -jnp.ones_like(event_colours))
            event_types = jnp.where(active, event_types, -jnp.ones_like(event_types))
        else:
            event_happened = None
            event_positions = None
            event_colours = None
            event_types = None
        return _TransitionProjection(
            state=self._project_state(timestep),
            action=jnp.where(active, action, jnp.asarray(-1, dtype=jnp.int32)),
            reward=reward,
            step_type=jnp.where(active, timestep.step_type, jnp.asarray(-1, dtype=jnp.int32)),
            done=done,
            active=active,
            event_happened=event_happened,
            event_positions=event_positions,
            event_colours=event_colours,
            event_types=event_types,
        )

    def _render_rgb(self, state: nx.State) -> jax.Array | None:
        """Render the configured RGB view when requested by the profile."""
        if self.profile.rgb_view_mode == "full":
            return nx.observations.rgb(state)
        if self.profile.rgb_view_mode == "first_person":
            return nx.observations.rgb_first_person(state)
        return None


def _project_events(
    events: nx.EventsManager,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    projected = tuple(_project_slot(events, keys) for _name, keys in NAVIX_EVENT_SLOT_SPECS)
    return (
        jnp.stack([item[0] for item in projected]).astype(jnp.bool_),
        jnp.stack([item[1] for item in projected]).astype(jnp.int32),
        jnp.stack([item[2] for item in projected]).astype(jnp.int32),
        jnp.stack([item[3] for item in projected]).astype(jnp.int32),
    )


def _project_slot(
    events: nx.EventsManager,
    keys: tuple[tuple[str, str], ...],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    happened = jnp.asarray(False)
    position = _ABSENT_POSITION
    colour = _ABSENT_SCALAR
    event_type = _ABSENT_SCALAR
    for key in keys:
        slot_happened, slot_position, slot_colour, slot_type = _project_key(events, key)
        take = slot_happened & jnp.logical_not(happened)
        position = jnp.where(take, slot_position, position)
        colour = jnp.where(take, slot_colour, colour)
        event_type = jnp.where(take, slot_type, event_type)
        happened = happened | slot_happened
    return happened, position, colour, event_type


def _project_key(
    events: nx.EventsManager,
    key: tuple[str, str],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    event = events.events.get(key)
    if event is None:
        return jnp.asarray(False), _ABSENT_POSITION, _ABSENT_SCALAR, _ABSENT_SCALAR
    happened_mask = jnp.asarray(event.happened, dtype=jnp.bool_)
    happened = jnp.any(happened_mask)
    positions = jnp.asarray(event.position, dtype=jnp.int32)
    colours = jnp.asarray(event.colour, dtype=jnp.int32)
    if positions.ndim == 1:
        chosen_position = positions
        chosen_colour = colours
    else:
        index = jnp.argmax(happened_mask.reshape(-1))
        chosen_position = positions.reshape(-1, 2)[index]
        chosen_colour = colours.reshape(-1)[index]
    type_code = jnp.asarray(NAVIX_EVENT_TYPE_CODES[key[1]], dtype=jnp.int32)
    chosen_position = jnp.where(happened, chosen_position, _ABSENT_POSITION)
    chosen_colour = jnp.where(happened, chosen_colour, _ABSENT_SCALAR)
    type_out = jnp.where(happened, type_code, _ABSENT_SCALAR)
    return happened, chosen_position, chosen_colour, type_out


def _project_mission(state: nx.State) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    mission = state.mission
    if len(mission) == 0:
        return (
            jnp.asarray(False),
            _ABSENT_POSITION,
            _ABSENT_SCALAR,
            _ABSENT_SCALAR,
        )
    target = mission[0]
    return (
        jnp.asarray(True),
        jnp.asarray(target.position, dtype=jnp.int32),
        jnp.asarray(target.colour, dtype=jnp.int32),
        _ABSENT_SCALAR,
    )


def _prepend_optional(initial: jax.Array | None, sequence: jax.Array | None) -> jax.Array | None:
    if initial is None or sequence is None:
        return None
    return jnp.concatenate([initial[None], sequence], axis=0)
