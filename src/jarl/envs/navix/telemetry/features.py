"""Visibility and dynamic-entity features derived from symbolic Navix states."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from jarl.envs.navix.telemetry._arrays import (
    bounded_items,
    episode_length,
    position,
    required_state_array,
)
from jarl.envs.navix.telemetry.contracts import (
    EntityTransition,
    NavixSummaryLimits,
    NavixTraceArrays,
    Position,
    StepInterval,
    SummarySectionCount,
    VisibilitySummary,
)

ENTITY_NAMES: dict[int, str] = {
    0: "unknown",
    1: "floor",
    2: "wall",
    4: "door",
    5: "key",
    6: "ball",
    7: "box",
    8: "goal",
    9: "lava",
    10: "player",
}
"""Navix symbolic entity tag names used in compact summaries."""

SALIENT_VISIBILITY_TAGS = frozenset({4, 5, 6, 7, 8, 9})
"""Entity tags summarized as task-relevant visibility evidence."""

DYNAMIC_ENTITY_TAGS = frozenset({6})
"""Entity tags whose occupancy changes are treated as autonomous movement."""

__all__ = [
    "DYNAMIC_ENTITY_TAGS",
    "ENTITY_NAMES",
    "SALIENT_VISIBILITY_TAGS",
    "entity_transitions",
    "format_step_intervals",
    "visibility_intervals",
]


def format_step_intervals(intervals: Sequence[StepInterval]) -> str:
    """Format inclusive intervals as comma-separated ranges."""
    return ", ".join(
        str(interval.start) if interval.start == interval.end else f"{interval.start}-{interval.end}"
        for interval in intervals
    )


def visibility_intervals(
    trace: NavixTraceArrays,
    *,
    episode_index: int = 0,
    limits: NavixSummaryLimits | None = None,
) -> tuple[VisibilitySummary, ...]:
    """Return visibility summaries for salient symbolic entities."""
    trace = trace.to_host()
    resolved_limits = limits or NavixSummaryLimits()
    length = episode_length(trace, episode_index)
    full_symbolic = required_state_array(trace.full_symbolic, "full_symbolic", episode_index, length)
    first_person = required_state_array(
        trace.first_person_symbolic,
        "first_person_symbolic",
        episode_index,
        length,
    )
    descriptors = _salient_descriptors(full_symbolic[0], first_person)
    summaries: list[VisibilitySummary] = []
    for entity_tag, colour, state in descriptors:
        visible_steps = [
            timestep
            for timestep, frame in enumerate(first_person)
            if np.any((frame[..., 0] == entity_tag) & (frame[..., 1] == colour) & (frame[..., 2] == state))
        ]
        intervals = _steps_to_intervals(visible_steps)
        returned_intervals = bounded_items(intervals, resolved_limits.max_visibility_intervals)
        summaries.append(
            VisibilitySummary(
                entity_tag=entity_tag,
                entity_name=_entity_name(entity_tag),
                colour=colour,
                state=state,
                ever_seen=bool(visible_steps),
                first_seen=visible_steps[0] if visible_steps else None,
                last_seen=visible_steps[-1] if visible_steps else None,
                visible_steps=len(visible_steps),
                interval_count=SummarySectionCount(
                    total=len(intervals),
                    returned=len(returned_intervals),
                ),
                intervals=tuple(returned_intervals),
                formatted_intervals=format_step_intervals(returned_intervals),
            )
        )
    return tuple(summaries)


def entity_transitions(
    trace: NavixTraceArrays,
    *,
    episode_index: int = 0,
) -> tuple[EntityTransition, ...]:
    """Return dynamic symbolic occupancy changes without inventing identity."""
    trace = trace.to_host()
    length = episode_length(trace, episode_index)
    full_symbolic = required_state_array(trace.full_symbolic, "full_symbolic", episode_index, length)
    transitions: list[EntityTransition] = []
    for timestep in range(length):
        before = full_symbolic[timestep]
        after = full_symbolic[timestep + 1]
        descriptors = _dynamic_descriptors(before, after)
        for entity_tag, colour, state in descriptors:
            before_positions = _positions_for_descriptor(before, entity_tag, colour, state)
            after_positions = _positions_for_descriptor(after, entity_tag, colour, state)
            removed = tuple(sorted(before_positions - after_positions))
            added = tuple(sorted(after_positions - before_positions))
            if not removed and not added:
                continue
            identity = (
                "unique"
                if len(before_positions) == 1 and len(after_positions) == 1 and len(removed) == 1 and len(added) == 1
                else "ambiguous"
            )
            transitions.append(
                EntityTransition(
                    timestep=timestep,
                    entity_tag=entity_tag,
                    entity_name=_entity_name(entity_tag),
                    colour=colour,
                    state=state,
                    from_positions=removed,
                    to_positions=added,
                    identity=identity,
                )
            )
    return tuple(transitions)


def _steps_to_intervals(steps: Sequence[int]) -> tuple[StepInterval, ...]:
    if not steps:
        return ()
    intervals: list[StepInterval] = []
    start = steps[0]
    end = start
    for step in steps[1:]:
        if step == end + 1:
            end = step
            continue
        intervals.append(StepInterval(start=start, end=end))
        start = step
        end = step
    intervals.append(StepInterval(start=start, end=end))
    return tuple(intervals)


def _salient_descriptors(initial_frame: np.ndarray, first_person: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    frames = (initial_frame, *first_person)
    descriptors = {
        (int(triple[0]), int(triple[1]), int(triple[2]))
        for frame in frames
        for triple in np.unique(frame.reshape(-1, 3), axis=0)
        if int(triple[0]) in SALIENT_VISIBILITY_TAGS
    }
    return tuple(sorted(descriptors))


def _dynamic_descriptors(before: np.ndarray, after: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    descriptors = {
        (int(triple[0]), int(triple[1]), int(triple[2]))
        for frame in (before, after)
        for triple in np.unique(frame.reshape(-1, 3), axis=0)
        if int(triple[0]) in DYNAMIC_ENTITY_TAGS
    }
    return tuple(sorted(descriptors))


def _positions_for_descriptor(
    frame: np.ndarray,
    entity_tag: int,
    colour: int,
    state: int,
) -> set[Position]:
    matches = np.argwhere((frame[..., 0] == entity_tag) & (frame[..., 1] == colour) & (frame[..., 2] == state))
    return {position(match) for match in matches}


def _entity_name(entity_tag: int) -> str:
    return ENTITY_NAMES.get(entity_tag, f"entity_{entity_tag}")
