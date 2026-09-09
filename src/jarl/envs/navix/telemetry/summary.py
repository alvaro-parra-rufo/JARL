"""Compact deterministic summaries for captured Navix episodes."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import pairwise

import numpy as np

from jarl.envs.navix.telemetry._arrays import (
    action_name,
    bounded_items,
    episode_length,
    position,
    required_state_array,
    transition_array,
)
from jarl.envs.navix.telemetry.contracts import (
    ActionSegment,
    ActionSummary,
    DerivedEvent,
    DynamicEntitySummary,
    EntityCount,
    EventCount,
    EventSummary,
    InitialStateSummary,
    NamedCount,
    NavixRolloutIdentity,
    NavixRolloutSummary,
    NavixSummaryLimits,
    NavixTraceArrays,
    OutcomeSummary,
    Position,
    SummarySectionCount,
    TrajectorySummary,
)
from jarl.envs.navix.telemetry.events import ROTATION_ACTION_NAMES, derive_events
from jarl.envs.navix.telemetry.features import (
    ENTITY_NAMES,
    entity_transitions,
    visibility_intervals,
)

_STEP_TYPE_TRUNCATION = 1
"""Navix step type value for time-limit truncation."""

_STEP_TYPE_TERMINATION = 2
"""Navix step type value for absorbing termination."""

__all__ = ["summarize_navix_trace"]


def summarize_navix_trace(
    trace: NavixTraceArrays,
    *,
    identity: NavixRolloutIdentity,
    action_names: Sequence[str],
    episode_index: int = 0,
    limits: NavixSummaryLimits | None = None,
) -> NavixRolloutSummary:
    """Build an exact aggregate and bounded-detail summary for one episode."""
    trace = trace.to_host()
    resolved_limits = limits or NavixSummaryLimits()
    length = episode_length(trace, episode_index)
    actions = transition_array(trace.actions, episode_index, length)
    positions = required_state_array(
        trace.player_positions,
        "player_positions",
        episode_index,
        length,
    )
    directions = required_state_array(
        trace.player_directions,
        "player_directions",
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
    events = derive_events(trace, action_names=action_names, episode_index=episode_index)
    transitions = entity_transitions(trace, episode_index=episode_index)
    visibility = visibility_intervals(trace, episode_index=episode_index, limits=resolved_limits)
    action_segments = _action_segments(actions, action_names)
    returned_action_segments = bounded_items(action_segments, resolved_limits.max_action_segments)
    returned_events = bounded_items(events, resolved_limits.max_events)
    returned_transitions = bounded_items(transitions, resolved_limits.max_entity_transitions)
    path = tuple(position(value) for value in positions)
    returned_path = bounded_items(path, resolved_limits.max_path_positions)

    return NavixRolloutSummary(
        identity=identity,
        outcome=_outcome(trace, events, episode_index=episode_index, length=length),
        initial_state=InitialStateSummary(
            map_shape=tuple(int(size) for size in full_symbolic[0].shape),
            map_hash=hashlib.sha256(full_symbolic[0].tobytes()).hexdigest()[:16],
            player_position=path[0],
            player_direction=int(directions[0]),
            player_pocket=int(pockets[0]),
            mission=_mission(trace, episode_index),
            entities=_entity_inventory(full_symbolic[0]),
        ),
        actions=ActionSummary(
            counts=_named_counts(action_name(int(action), action_names) for action in actions),
            segment_count=SummarySectionCount(
                total=len(action_segments),
                returned=len(returned_action_segments),
            ),
            segments=tuple(returned_action_segments),
        ),
        events=EventSummary(
            counts=_event_counts(events),
            timeline_count=SummarySectionCount(
                total=len(events),
                returned=len(returned_events),
            ),
            timeline=tuple(returned_events),
        ),
        visibility=visibility,
        trajectory=TrajectorySummary(
            start=path[0],
            end=path[-1],
            unique_cells=len(set(path)),
            manhattan_distance=sum(_manhattan(left, right) for left, right in pairwise(path)),
            rotations=sum(action_name(int(action), action_names) in ROTATION_ACTION_NAMES for action in actions),
            blocked_actions=sum(event.kind == "movement_blocked" for event in events),
            path_count=SummarySectionCount(total=len(path), returned=len(returned_path)),
            path=tuple(returned_path),
        ),
        dynamic_entities=DynamicEntitySummary(
            transition_count=SummarySectionCount(
                total=len(transitions),
                returned=len(returned_transitions),
            ),
            transitions=tuple(returned_transitions),
        ),
    )


def _outcome(
    trace: NavixTraceArrays,
    events: Sequence[DerivedEvent],
    *,
    episode_index: int,
    length: int,
) -> OutcomeSummary:
    step_types = transition_array(trace.step_types, episode_index, length)
    final_step_type = int(step_types[-1]) if length else -1
    terminated = final_step_type == _STEP_TYPE_TERMINATION
    truncated = final_step_type == _STEP_TYPE_TRUNCATION
    event_kinds = {event.kind for event in events}
    if "goal_reached" in event_kinds:
        reason = "goal_reached"
        success: bool | None = True
    elif "lava_fall" in event_kinds:
        reason = "lava_fall"
        success = False
    elif "ball_hit" in event_kinds:
        reason = "ball_hit"
        success = False
    elif terminated:
        reason = "termination"
        success = None
    elif truncated:
        reason = "truncation"
        success = None
    else:
        reason = "horizon"
        success = None
    returns = np.asarray(trace.episode_returns)
    return OutcomeSummary(
        return_total=float(returns[episode_index]),
        length=length,
        terminated=terminated,
        truncated=truncated,
        success=success,
        reason=reason,
    )


def _mission(trace: NavixTraceArrays, episode_index: int) -> DerivedEvent | None:
    present = bool(np.asarray(trace.mission_present)[episode_index])
    if not present:
        return None
    mission_position = position(np.asarray(trace.mission_position)[episode_index])
    colour = int(np.asarray(trace.mission_colour)[episode_index])
    event_type = int(np.asarray(trace.mission_event_type)[episode_index])
    return DerivedEvent(
        timestep=0,
        kind="mission",
        source="navix",
        position=mission_position,
        colour=None if colour < 0 else colour,
        event_type=None if event_type < 0 else event_type,
        slot="mission",
    )


def _entity_inventory(frame: np.ndarray) -> tuple[EntityCount, ...]:
    triples, counts = np.unique(frame.reshape(-1, 3), axis=0, return_counts=True)
    entities = [
        EntityCount(
            entity_tag=int(triple[0]),
            entity_name=ENTITY_NAMES.get(int(triple[0]), f"entity_{int(triple[0])}"),
            colour=int(triple[1]),
            state=int(triple[2]),
            count=int(count),
        )
        for triple, count in zip(triples, counts, strict=True)
        if int(triple[0]) not in {0, 1}
    ]
    return tuple(sorted(entities, key=lambda entity: (entity.entity_tag, entity.colour, entity.state)))


def _action_segments(actions: np.ndarray, action_names: Sequence[str]) -> tuple[ActionSegment, ...]:
    if len(actions) == 0:
        return ()
    segments: list[ActionSegment] = []
    start = 0
    current = int(actions[0])
    for timestep in range(1, len(actions)):
        action = int(actions[timestep])
        if action == current:
            continue
        segments.append(
            ActionSegment(
                start=start,
                end=timestep - 1,
                action_index=current,
                action_name=action_name(current, action_names),
            )
        )
        start = timestep
        current = action
    segments.append(
        ActionSegment(
            start=start,
            end=len(actions) - 1,
            action_index=current,
            action_name=action_name(current, action_names),
        )
    )
    return tuple(segments)


def _manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _named_counts(names: Iterable[str]) -> tuple[NamedCount, ...]:
    counts = Counter(names)
    return tuple(NamedCount(name=name, count=count) for name, count in sorted(counts.items()))


def _event_counts(events: Iterable[DerivedEvent]) -> tuple[EventCount, ...]:
    counts = Counter((event.kind, event.source) for event in events)
    return tuple(EventCount(kind=kind, source=source, count=count) for (kind, source), count in sorted(counts.items()))
