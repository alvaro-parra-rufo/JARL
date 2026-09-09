"""Configurable Navix reward primitives and occupancy helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import navix as nx
from jax import Array
from navix.components import DISCARD_PILE_COORDS, EMPTY_POCKET_ID
from navix.entities import Entities, EntityIds
from navix.grid import crop, translate
from navix.observations import RADIUS, symbolic
from navix.states import EventType, State

from jarl.envs.navix.reward_config import RewardWeightsConfig

__all__ = [
    "CHANNEL_SPECS",
    "ChannelSpec",
    "RewardFn",
    "any_door_open_reward",
    "channel_catalog",
    "closed_door_proximity_reward",
    "compose_reward_fn",
    "done_at_goal_reward",
    "door_done_reward",
    "entity_in_fov",
    "event_reward",
    "facing_entity",
    "goal_approach_reward",
    "grid_d_max",
    "holding_key_reward",
    "lava_clearance_reward",
    "manhattan",
    "nearest_proximity_reward",
    "occupancy_enter_reward",
    "occupancy_entries",
    "occupancy_mask",
    "occupancy_reward",
    "occupancy_steps",
    "on_map_mask",
    "scale_reward",
]

RewardFn = Callable[[State, Array, State], Array]
"""Navix reward callable: ``(prev_state, action, state) -> f32[]``."""

_DONE_ACTION = 6
"""Discrete index of the Navix ``done`` action."""

_DISCARD = jnp.asarray(DISCARD_PILE_COORDS, dtype=jnp.int32)

_ENTITY_IDS: dict[str, Array] = {
    Entities.GOAL: EntityIds.GOAL,
    Entities.KEY: EntityIds.KEY,
    Entities.DOOR: EntityIds.DOOR,
    Entities.LAVA: EntityIds.LAVA,
    Entities.BALL: EntityIds.BALL,
    Entities.BOX: EntityIds.BOX,
}


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """One configurable reward channel.

    Args:
        name: Field name on ``RewardWeightsConfig``.
        kind: Compact label for the channel (event kinds may differ from ``name``).
        primitive: Channel function producing ``c_i`` in ``[0, 1]``.
    """

    name: str
    kind: str
    primitive: RewardFn


def grid_d_max(state: State) -> Array:
    """Return ``(H - 1) + (W - 1)`` as a positive ``float32`` scalar."""
    height = jnp.asarray(state.grid.shape[0], dtype=jnp.float32)
    width = jnp.asarray(state.grid.shape[1], dtype=jnp.float32)
    return jnp.maximum((height - 1.0) + (width - 1.0), 1.0)


def manhattan(a: Array, b: Array) -> Array:
    """Return Manhattan distance between two ``i32[2]`` coordinates."""
    delta = jnp.abs(jnp.asarray(a, dtype=jnp.int32) - jnp.asarray(b, dtype=jnp.int32))
    return jnp.asarray(jnp.sum(delta), dtype=jnp.float32)


def on_map_mask(positions: Array) -> Array:
    """Return a boolean mask that is false on the Navix discard pile."""
    coords = jnp.asarray(positions, dtype=jnp.int32)
    if coords.ndim == 1:
        coords = coords[None]
    return ~jnp.all(coords == _DISCARD, axis=-1)


def scale_reward(fn: RewardFn, weight: float) -> RewardFn:
    """Return ``fn`` multiplied by a non-negative scalar weight."""
    scale = jnp.asarray(weight, dtype=jnp.float32)

    def scaled(prev_state: State, action: Array, state: State) -> Array:
        return scale * jnp.asarray(fn(prev_state, action, state), dtype=jnp.float32)

    return scaled


def event_reward(key: tuple[str, str]) -> RewardFn:
    """Return a 1/0 pulse when ``key`` fires this step.

    ``key`` is a Navix ``(entity_type, event_type)`` pair. Maps that never
    construct that entity yield 0.
    """

    def pulse(prev_state: State, action: Array, state: State) -> Array:
        del prev_state, action
        return jnp.asarray(state.events.happened(key), dtype=jnp.float32)

    return pulse


def holding_key_reward(prev_state: State, action: Array, state: State) -> Array:
    """Return 1 when the agent pocket is not empty."""
    del prev_state, action
    pocket = jnp.asarray(state.get_player().pocket)
    return jnp.asarray(pocket != EMPTY_POCKET_ID, dtype=jnp.float32)


def any_door_open_reward(prev_state: State, action: Array, state: State) -> Array:
    """Return 1 when at least one door is open."""
    del prev_state, action
    if Entities.DOOR not in state.entities:
        return jnp.asarray(0.0, dtype=jnp.float32)
    opened = jnp.asarray(state.get_doors().open, dtype=jnp.bool_)
    return jnp.asarray(jnp.any(opened), dtype=jnp.float32)


def nearest_proximity_reward(entity_enum: str) -> RewardFn:
    """Return ``1 - min_d / d_max`` to the nearest on-map entity of ``entity_enum``."""

    def proximity(prev_state: State, action: Array, state: State) -> Array:
        del prev_state, action
        has, min_d = _nearest_on_map_distance(state, entity_enum)
        closeness = 1.0 - (min_d / grid_d_max(state))
        return jnp.where(has, jnp.clip(closeness, 0.0, 1.0), jnp.asarray(0.0, dtype=jnp.float32))

    return proximity


def closed_door_proximity_reward(prev_state: State, action: Array, state: State) -> Array:
    """Return proximity to the nearest closed on-map door."""
    del prev_state, action
    if Entities.DOOR not in state.entities:
        return jnp.asarray(0.0, dtype=jnp.float32)
    doors = state.get_doors()
    closed = jnp.logical_not(jnp.asarray(doors.open, dtype=jnp.bool_))
    has, min_d = _nearest_on_map_distance(state, Entities.DOOR, extra_mask=closed)
    closeness = 1.0 - (min_d / grid_d_max(state))
    return jnp.where(has, jnp.clip(closeness, 0.0, 1.0), jnp.asarray(0.0, dtype=jnp.float32))


def lava_clearance_reward(prev_state: State, action: Array, state: State) -> Array:
    """Return normalized distance to the nearest on-map lava."""
    del prev_state, action
    has, min_d = _nearest_on_map_distance(state, Entities.LAVA)
    clearance = min_d / grid_d_max(state)
    return jnp.where(has, jnp.clip(clearance, 0.0, 1.0), jnp.asarray(0.0, dtype=jnp.float32))


def entity_in_fov(entity_enum: str) -> RewardFn:
    """Return 1 when an on-map entity of ``entity_enum`` is in the first-person crop."""
    entity_id = _ENTITY_IDS[entity_enum]

    def visible(prev_state: State, action: Array, state: State) -> Array:
        del prev_state, action
        if entity_enum not in state.entities:
            return jnp.asarray(0.0, dtype=jnp.float32)
        player = state.get_player()
        tags = symbolic(state)[..., 0]
        cropped = crop(tags, player.position, player.direction, RADIUS, padding_value=255)
        present = jnp.any(jnp.equal(cropped, entity_id))
        return jnp.asarray(present, dtype=jnp.float32)

    return visible


def facing_entity(entity_enum: str) -> RewardFn:
    """Return 1 when the cell in front of the agent holds an on-map entity."""

    def facing(prev_state: State, action: Array, state: State) -> Array:
        del prev_state, action
        if entity_enum not in state.entities:
            return jnp.asarray(0.0, dtype=jnp.float32)
        player = state.get_player()
        forward = translate(player.position, player.direction)
        positions = jnp.asarray(state.entities[entity_enum].position, dtype=jnp.int32)
        if positions.ndim == 1:
            positions = positions[None]
        matches = jnp.all(positions == forward, axis=-1) & on_map_mask(positions)
        return jnp.asarray(jnp.any(matches), dtype=jnp.float32)

    return facing


def done_at_goal_reward(prev_state: State, action: Array, state: State) -> Array:
    """Return 1 when ``done`` is taken while occupying or facing a goal."""
    del prev_state
    is_done = jnp.equal(jnp.asarray(action, dtype=jnp.int32), _DONE_ACTION)
    on_goal = _occupies_entity(state, Entities.GOAL)
    facing_goal = facing_entity(Entities.GOAL)(state, action, state)
    return jnp.asarray(is_done & (on_goal | jnp.asarray(facing_goal, dtype=jnp.bool_)), dtype=jnp.float32)


def door_done_reward(prev_state: State, action: Array, state: State) -> Array:
    """Return the Navix door-done indicator without asserting a mission."""
    del prev_state, action
    if not state.mission or Entities.DOOR not in state.entities:
        return jnp.asarray(0.0, dtype=jnp.float32)
    player = state.get_player()
    forward = translate(player.position, player.direction)
    doors = state.get_doors()
    idx = jnp.where(nx.grid.positions_equal(doors.position, forward), size=1)[0][0]
    door = doors[idx]
    target = state.mission[0]
    pos_match = jnp.array_equal(forward, target.position)
    colour_match = jnp.array_equal(door.colour, target.colour)
    return jnp.asarray(jnp.logical_and(pos_match, colour_match), dtype=jnp.float32)


def goal_approach_reward(prev_state: State, action: Array, state: State) -> Array:
    """Return the clipped reduction in Manhattan distance to the nearest goal."""
    del action
    has_prev, d_prev = _nearest_on_map_distance(prev_state, Entities.GOAL)
    has_curr, d_curr = _nearest_on_map_distance(state, Entities.GOAL)
    delta = jnp.clip(d_prev - d_curr, 0.0, 1.0)
    return jnp.where(has_prev & has_curr, delta, jnp.asarray(0.0, dtype=jnp.float32))


def occupancy_enter_reward(cell_pos: tuple[int, int] | Array) -> RewardFn:
    """Return a pulse when the agent enters ``cell_pos`` this step."""
    cell = jnp.asarray(cell_pos, dtype=jnp.int32)

    def enter(prev_state: State, action: Array, state: State) -> Array:
        del action
        was = jnp.all(jnp.asarray(prev_state.get_player().position, dtype=jnp.int32) == cell)
        now = jnp.all(jnp.asarray(state.get_player().position, dtype=jnp.int32) == cell)
        return jnp.asarray(jnp.logical_and(jnp.logical_not(was), now), dtype=jnp.float32)

    return enter


def occupancy_reward(cell_pos: tuple[int, int] | Array) -> RewardFn:
    """Return 1 when the agent occupies ``cell_pos``."""
    cell = jnp.asarray(cell_pos, dtype=jnp.int32)

    def occupied(prev_state: State, action: Array, state: State) -> Array:
        del prev_state, action
        now = jnp.all(jnp.asarray(state.get_player().position, dtype=jnp.int32) == cell)
        return jnp.asarray(now, dtype=jnp.float32)

    return occupied


def occupancy_mask(positions: Array, cell_pos: tuple[int, int] | Array) -> Array:
    """Return a boolean mask of timesteps spent on ``cell_pos``."""
    coords = jnp.asarray(positions, dtype=jnp.int32)
    cell = jnp.asarray(cell_pos, dtype=jnp.int32)
    return jnp.all(coords == cell, axis=-1)


def occupancy_steps(mask: Array) -> int:
    """Return how many timesteps the occupancy mask is true."""
    return int(jnp.asarray(mask).sum())


def occupancy_entries(mask: Array) -> int:
    """Return rising-edge counts ``False -> True`` on the occupancy mask."""
    flags = jnp.asarray(mask, dtype=jnp.bool_)
    if flags.size == 0:
        return 0
    previous = jnp.concatenate([jnp.asarray([False]), flags[:-1]])
    return int(jnp.logical_and(flags, jnp.logical_not(previous)).sum())


def compose_reward_fn(weights: RewardWeightsConfig) -> RewardFn:
    """Return the weighted sum of the 20 configurable channels."""
    payload = weights.model_dump()
    scaled = tuple(scale_reward(spec.primitive, float(payload[spec.name])) for spec in CHANNEL_SPECS)
    return nx.rewards.compose(*scaled, operator=jnp.sum)


def channel_catalog() -> tuple[ChannelSpec, ...]:
    """Return the configurable channel catalog in schema order."""
    return CHANNEL_SPECS


def _entity_positions(state: State, entity_enum: str) -> Array | None:
    if entity_enum not in state.entities:
        return None
    positions = jnp.asarray(state.entities[entity_enum].position, dtype=jnp.int32)
    if positions.ndim == 1:
        return positions[None]
    return positions


def _nearest_on_map_distance(
    state: State,
    entity_enum: str,
    extra_mask: Array | None = None,
) -> tuple[Array, Array]:
    positions = _entity_positions(state, entity_enum)
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    if positions is None:
        return jnp.asarray(False), zero
    mask = on_map_mask(positions)
    if extra_mask is not None:
        mask = mask & jnp.asarray(extra_mask, dtype=jnp.bool_)
    player = jnp.asarray(state.get_player().position, dtype=jnp.int32)
    distances = jnp.sum(jnp.abs(positions - player), axis=-1).astype(jnp.float32)
    finite = jnp.where(mask, distances, jnp.asarray(jnp.inf, dtype=jnp.float32))
    return jnp.any(mask), jnp.min(finite)


def _occupies_entity(state: State, entity_enum: str) -> Array:
    positions = _entity_positions(state, entity_enum)
    if positions is None:
        return jnp.asarray(False)
    player = jnp.asarray(state.get_player().position, dtype=jnp.int32)
    matches = jnp.all(positions == player, axis=-1) & on_map_mask(positions)
    return jnp.any(matches)


CHANNEL_SPECS: tuple[ChannelSpec, ...] = (
    ChannelSpec("goal_reached", "goal_reached", event_reward((Entities.GOAL, EventType.REACH))),
    ChannelSpec("key_pickup", "key_pickup", event_reward((Entities.KEY, EventType.PICKUP))),
    ChannelSpec("door_opening", "door_open", event_reward((Entities.DOOR, EventType.OPEN))),
    ChannelSpec("door_unlock", "door_unlock", event_reward((Entities.DOOR, EventType.UNLOCK))),
    ChannelSpec("ball_pickup", "ball_pickup", event_reward((Entities.BALL, EventType.PICKUP))),
    ChannelSpec("box_pickup", "box_pickup", event_reward((Entities.BOX, EventType.PICKUP))),
    ChannelSpec("door_done", "door_done", door_done_reward),
    ChannelSpec("holding_key", "holding_key", holding_key_reward),
    ChannelSpec("door_is_open", "door_is_open", any_door_open_reward),
    ChannelSpec("distance_to_goal", "distance_to_goal", nearest_proximity_reward(Entities.GOAL)),
    ChannelSpec("distance_to_key", "distance_to_key", nearest_proximity_reward(Entities.KEY)),
    ChannelSpec("distance_to_door", "distance_to_door", closed_door_proximity_reward),
    ChannelSpec("lava_clearance", "lava_clearance", lava_clearance_reward),
    ChannelSpec("distance_to_ball", "distance_to_ball", nearest_proximity_reward(Entities.BALL)),
    ChannelSpec("distance_to_box", "distance_to_box", nearest_proximity_reward(Entities.BOX)),
    ChannelSpec("goal_visible", "goal_visible", entity_in_fov(Entities.GOAL)),
    ChannelSpec("key_visible", "key_visible", entity_in_fov(Entities.KEY)),
    ChannelSpec("facing_goal", "facing_goal", facing_entity(Entities.GOAL)),
    ChannelSpec("facing_key", "facing_key", facing_entity(Entities.KEY)),
    ChannelSpec("done_at_goal", "done_at_goal", done_at_goal_reward),
    ChannelSpec("goal_approach", "goal_approach", goal_approach_reward),
)
"""Catalog of the 21 configurable reward channels, excluding scenario overlays."""
