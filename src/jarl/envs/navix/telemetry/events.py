"""Physical event normalization for Navix rollout traces."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from jarl.envs.navix.telemetry._arrays import (
    action_name,
    episode_length,
    player_direction,
    position,
    required_state_array,
    required_transition_array,
    transition_array,
    validate_action_indices,
)
from jarl.envs.navix.telemetry.contracts import (
    NAVIX_EVENT_SLOT_NAMES,
    NAVIX_EVENT_TYPE_CODES,
    DerivedEvent,
    NavixTraceArrays,
    Position,
)

MOVEMENT_ACTION_NAMES = frozenset({"forward", "backward"})
"""Navix actions expected to change player position when unblocked."""

ROTATION_ACTION_NAMES = frozenset({"rotate_cw", "rotate_ccw", "left", "right"})
"""In-place turns. Includes MiniGrid ``left``/``right`` labels from older traces."""

_DOOR_STATE_LOCKED = 2
"""Navix symbolic state value for a closed and locked door."""

__all__ = [
    "MOVEMENT_ACTION_NAMES",
    "ROTATION_ACTION_NAMES",
    "derive_events",
]


def derive_events(
    trace: NavixTraceArrays,
    *,
    action_names: Sequence[str],
    episode_index: int = 0,
) -> tuple[DerivedEvent, ...]:
    """Normalize Navix event slots and unambiguous state-difference events."""
    trace = trace.to_host()
    length = episode_length(trace, episode_index)
    actions = transition_array(trace.actions, episode_index, length)
    validate_action_indices(actions, action_names)
    positions = required_state_array(
        trace.player_positions,
        "player_positions",
        episode_index,
        length,
    )
    pockets = required_state_array(
        trace.player_pockets,
        "player_pockets",
        episode_index,
        length,
    )
    full_symbolic = required_state_array(trace.full_symbolic, "full_symbolic", episode_index, length)
    events = list(_raw_events(trace, episode_index=episode_index, length=length))
    known = {(event.timestep, event.kind, event.position) for event in events}

    for timestep in range(length):
        resolved_action_name = action_name(int(actions[timestep]), action_names)
        current_position = position(positions[timestep])
        next_position = position(positions[timestep + 1])
        if resolved_action_name in MOVEMENT_ACTION_NAMES and current_position == next_position:
            _append_derived_event(
                events,
                known,
                DerivedEvent(
                    timestep=timestep,
                    kind="movement_blocked",
                    source="derived",
                    position=current_position,
                ),
            )
        if resolved_action_name == "pickup" and int(pockets[timestep]) != int(pockets[timestep + 1]):
            _append_derived_event(
                events,
                known,
                DerivedEvent(
                    timestep=timestep,
                    kind="key_pickup",
                    source="derived",
                    position=_position_in_front(
                        current_position,
                        player_direction(trace, episode_index, timestep),
                    ),
                ),
            )
        for event in _door_events(
            full_symbolic[timestep],
            full_symbolic[timestep + 1],
            timestep=timestep,
        ):
            _append_derived_event(events, known, event)

    return tuple(sorted(events, key=lambda event: (event.timestep, event.source != "navix", event.kind)))


def _raw_events(
    trace: NavixTraceArrays,
    *,
    episode_index: int,
    length: int,
) -> tuple[DerivedEvent, ...]:
    happened = required_transition_array(
        trace.event_happened,
        "event_happened",
        episode_index,
        length,
    )
    positions = required_transition_array(
        trace.event_positions,
        "event_positions",
        episode_index,
        length,
    )
    colours = required_transition_array(
        trace.event_colours,
        "event_colours",
        episode_index,
        length,
    )
    event_types = required_transition_array(
        trace.event_types,
        "event_types",
        episode_index,
        length,
    )
    previous: list[tuple[Position, int, int] | None] = [None] * len(NAVIX_EVENT_SLOT_NAMES)
    events: list[DerivedEvent] = []
    for timestep in range(length):
        for slot_index, slot_name in enumerate(NAVIX_EVENT_SLOT_NAMES):
            if not bool(happened[timestep, slot_index]):
                previous[slot_index] = None
                continue
            event_position = position(positions[timestep, slot_index])
            colour = int(colours[timestep, slot_index])
            event_type = int(event_types[timestep, slot_index])
            signature = (event_position, colour, event_type)
            if previous[slot_index] == signature:
                continue
            previous[slot_index] = signature
            persists_identically = (
                timestep + 1 < length
                and bool(happened[timestep + 1, slot_index])
                and position(positions[timestep + 1, slot_index]) == event_position
                and int(colours[timestep + 1, slot_index]) == colour
                and int(event_types[timestep + 1, slot_index]) == event_type
            )
            events.append(
                DerivedEvent(
                    timestep=timestep,
                    kind=_event_kind(slot_name, event_type),
                    source="navix",
                    position=None if event_position == (-1, -1) else event_position,
                    colour=None if colour < 0 else colour,
                    event_type=None if event_type < 0 else event_type,
                    slot=slot_name,
                    ambiguous=persists_identically,
                )
            )
    return tuple(events)


def _door_events(
    before: np.ndarray,
    after: np.ndarray,
    *,
    timestep: int,
) -> tuple[DerivedEvent, ...]:
    before_doors = _descriptor_by_position(before, entity_tag=4)
    after_doors = _descriptor_by_position(after, entity_tag=4)
    events: list[DerivedEvent] = []
    for door_position in sorted(before_doors.keys() & after_doors.keys()):
        _before_colour, before_state = before_doors[door_position]
        after_colour, after_state = after_doors[door_position]
        if before_state == _DOOR_STATE_LOCKED and after_state < _DOOR_STATE_LOCKED:
            events.append(
                DerivedEvent(
                    timestep=timestep,
                    kind="door_unlock",
                    source="derived",
                    position=door_position,
                    colour=after_colour,
                )
            )
        if before_state != 0 and after_state == 0:
            events.append(
                DerivedEvent(
                    timestep=timestep,
                    kind="door_open",
                    source="derived",
                    position=door_position,
                    colour=after_colour,
                )
            )
    return tuple(events)


def _descriptor_by_position(frame: np.ndarray, *, entity_tag: int) -> dict[Position, tuple[int, int]]:
    matches = np.argwhere(frame[..., 0] == entity_tag)
    return {
        position(match): (
            int(frame[tuple(match)][1]),
            int(frame[tuple(match)][2]),
        )
        for match in matches
    }


def _event_kind(slot_name: str, event_type: int) -> str:
    if event_type == NAVIX_EVENT_TYPE_CODES["unlock"]:
        return "door_unlock"
    if event_type == NAVIX_EVENT_TYPE_CODES["open"]:
        return "door_open"
    return {
        "goal_reached": "goal_reached",
        "ball_hit": "ball_hit",
        "wall_hit": "wall_hit",
        "lava_fall": "lava_fall",
        "key_pickup": "key_pickup",
        "door_opening": "door_open",
        "door_unlock": "door_unlock",
        "ball_pickup": "ball_pickup",
        "box_pickup": "box_pickup",
    }[slot_name]


def _append_derived_event(
    events: list[DerivedEvent],
    known: set[tuple[int, str, Position | None]],
    event: DerivedEvent,
) -> None:
    key = (event.timestep, event.kind, event.position)
    if key in known:
        return
    known.add(key)
    events.append(event)


def _position_in_front(player_position: Position, direction: int) -> Position:
    offsets = ((0, 1), (1, 0), (0, -1), (-1, 0))
    row_offset, column_offset = offsets[direction % len(offsets)]
    return player_position[0] + row_offset, player_position[1] + column_offset
