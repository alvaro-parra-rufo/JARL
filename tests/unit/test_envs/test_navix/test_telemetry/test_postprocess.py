"""Tests for deterministic Navix telemetry postprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from jarl.envs.navix.telemetry import (
    NAVIX_EVENT_SLOT_NAMES,
    NAVIX_EVENT_TYPE_CODES,
    NavixRolloutIdentity,
    NavixSummaryLimits,
    NavixTraceArrays,
    derive_events,
    entity_transitions,
    format_step_intervals,
    summarize_navix_trace,
    visibility_intervals,
)

_ACTION_NAMES = (
    "rotate_ccw",
    "rotate_cw",
    "forward",
    "pickup",
    "drop",
    "toggle",
    "done",
)


class TestNavixPostprocessing:
    """State and event arrays become exact aggregates with bounded detail."""

    def test_derive_events_separates_raw_flags_and_state_diffs(self) -> None:
        trace = _trace()

        events = derive_events(trace, action_names=_ACTION_NAMES)

        assert sum(event.kind == "wall_hit" and event.source == "navix" for event in events) == 1
        wall_event = next(event for event in events if event.kind == "wall_hit")
        assert wall_event.ambiguous is True
        assert any(event.kind == "movement_blocked" and event.timestep == 1 for event in events)
        assert any(event.kind == "key_pickup" and event.source == "derived" for event in events)
        assert any(event.kind == "door_unlock" and event.source == "navix" for event in events)
        assert any(event.kind == "door_open" and event.source == "derived" for event in events)
        assert any(event.kind == "goal_reached" and event.source == "navix" for event in events)

    def test_visibility_preserves_exact_totals_and_formats_ranges(self) -> None:
        trace = _trace()

        visibility = visibility_intervals(trace)
        key_visibility = next(item for item in visibility if item.entity_name == "key")

        assert key_visibility.visible_steps == 3
        assert key_visibility.first_seen == 1
        assert key_visibility.last_seen == 4
        assert format_step_intervals(key_visibility.intervals) == "1-2, 4"
        assert key_visibility.formatted_intervals == "1-2, 4"

    def test_dynamic_entity_transition_is_unique_for_single_ball(self) -> None:
        transitions = entity_transitions(_trace())

        ball_transition = next(item for item in transitions if item.entity_name == "ball")
        assert ball_transition.timestep == 0
        assert ball_transition.from_positions == ((2, 1),)
        assert ball_transition.to_positions == ((2, 2),)
        assert ball_transition.identity == "unique"

    def test_multiple_indistinguishable_entities_remain_ambiguous(self) -> None:
        trace = _trace(two_balls=True)

        transitions = entity_transitions(trace)

        assert transitions
        assert all(item.identity == "ambiguous" for item in transitions)

    def test_summary_limits_only_detail_not_exact_aggregates(self) -> None:
        trace = _trace()
        limits = NavixSummaryLimits(
            max_action_segments=2,
            max_events=2,
            max_visibility_intervals=1,
            max_entity_transitions=1,
            max_path_positions=3,
        )

        summary = summarize_navix_trace(
            trace,
            identity=NavixRolloutIdentity(env_id="Navix-Test-v0", seed=7),
            action_names=_ACTION_NAMES,
            limits=limits,
        )

        assert summary.outcome.return_total == 1.0
        assert summary.outcome.length == 5
        assert summary.outcome.terminated is True
        assert summary.outcome.success is True
        assert summary.actions.segment_count.total == 4
        assert summary.actions.segment_count.returned == 2
        assert summary.actions.segment_count.truncated is True
        assert sum(item.count for item in summary.actions.counts) == 5
        assert summary.events.timeline_count.total > summary.events.timeline_count.returned
        assert summary.events.timeline_count.returned == 2
        assert any(
            count.kind == "wall_hit" and count.source == "navix" and count.count == 1 for count in summary.events.counts
        )
        assert any(
            count.kind == "movement_blocked" and count.source == "derived" and count.count == 1
            for count in summary.events.counts
        )
        key_visibility = next(item for item in summary.visibility if item.entity_name == "key")
        assert key_visibility.interval_count.total == 2
        assert key_visibility.interval_count.returned == 1
        assert key_visibility.interval_count.truncated is True
        assert summary.trajectory.unique_cells == 3
        assert summary.trajectory.manhattan_distance == 2
        assert summary.trajectory.blocked_actions == 1
        assert summary.trajectory.path_count.total == 6
        assert summary.trajectory.path_count.returned == 3
        assert summary.initial_state.map_hash
        assert summary.initial_state.mission is not None

    def test_lava_termination_is_not_reported_as_success(self) -> None:
        trace = _trace(outcome_event="lava")

        summary = summarize_navix_trace(
            trace,
            identity=NavixRolloutIdentity(env_id="Navix-Test-v0"),
            action_names=_ACTION_NAMES,
        )

        assert summary.outcome.reason == "lava_fall"
        assert summary.outcome.success is False

    def test_unknown_termination_does_not_invent_success_semantics(self) -> None:
        trace = _trace(outcome_event="none")

        summary = summarize_navix_trace(
            trace,
            identity=NavixRolloutIdentity(env_id="Navix-Test-v0"),
            action_names=_ACTION_NAMES,
        )

        assert summary.outcome.reason == "termination"
        assert summary.outcome.success is None

    def test_action_indices_are_validated_against_environment_names(self) -> None:
        with pytest.raises(ValueError, match="outside the environment action set"):
            derive_events(_trace(), action_names=("rotate_ccw",))


def _trace(
    *,
    two_balls: bool = False,
    outcome_event: str = "goal",
) -> NavixTraceArrays:
    states = np.stack([_symbolic_state(timestep, two_balls=two_balls) for timestep in range(6)])
    first_person = np.ones((6, 3, 3, 3), dtype=np.int32)
    for timestep in (1, 2, 4):
        first_person[timestep, 1, 1] = np.asarray([5, 2, 0], dtype=np.int32)
    n_slots = len(NAVIX_EVENT_SLOT_NAMES)
    event_happened = np.zeros((5, n_slots), dtype=np.bool_)
    event_positions = -np.ones((5, n_slots, 2), dtype=np.int32)
    event_colours = -np.ones((5, n_slots), dtype=np.int32)
    event_types = -np.ones((5, n_slots), dtype=np.int32)
    _set_event(
        event_happened,
        event_positions,
        event_colours,
        event_types,
        timestep=1,
        slot=NAVIX_EVENT_SLOT_NAMES.index("wall_hit"),
        position=(1, 3),
        colour=0,
        event_type=NAVIX_EVENT_TYPE_CODES["hit"],
    )
    _set_event(
        event_happened,
        event_positions,
        event_colours,
        event_types,
        timestep=2,
        slot=NAVIX_EVENT_SLOT_NAMES.index("wall_hit"),
        position=(1, 3),
        colour=0,
        event_type=NAVIX_EVENT_TYPE_CODES["hit"],
    )
    _set_event(
        event_happened,
        event_positions,
        event_colours,
        event_types,
        timestep=3,
        slot=NAVIX_EVENT_SLOT_NAMES.index("door_unlock"),
        position=(1, 3),
        colour=3,
        event_type=NAVIX_EVENT_TYPE_CODES["unlock"],
    )
    if outcome_event == "goal":
        _set_event(
            event_happened,
            event_positions,
            event_colours,
            event_types,
            timestep=4,
            slot=NAVIX_EVENT_SLOT_NAMES.index("goal_reached"),
            position=(2, 2),
            colour=0,
            event_type=NAVIX_EVENT_TYPE_CODES["reach"],
        )
    elif outcome_event == "lava":
        _set_event(
            event_happened,
            event_positions,
            event_colours,
            event_types,
            timestep=4,
            slot=NAVIX_EVENT_SLOT_NAMES.index("lava_fall"),
            position=(2, 2),
            colour=0,
            event_type=NAVIX_EVENT_TYPE_CODES["fall"],
        )
    elif outcome_event != "none":
        msg = f"Unsupported outcome event: {outcome_event!r}."
        raise ValueError(msg)
    return NavixTraceArrays(
        full_symbolic=states[None],
        first_person_symbolic=first_person[None],
        policy_observations=np.zeros((1, 6, 27), dtype=np.float32),
        player_positions=np.asarray(
            [[(1, 1), (1, 2), (1, 2), (1, 2), (1, 2), (2, 2)]],
            dtype=np.int32,
        ),
        player_directions=np.zeros((1, 6), dtype=np.int32),
        player_pockets=np.asarray([[-1, -1, -1, 0, 0, 0]], dtype=np.int32),
        rgb_frames=None,
        actions=np.asarray([[2, 2, 3, 5, 2]], dtype=np.int32),
        rewards=np.asarray([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        step_types=np.asarray([[0, 0, 0, 0, 2]], dtype=np.int32),
        dones=np.asarray([[False, False, False, False, True]], dtype=np.bool_),
        active_mask=np.ones((1, 5), dtype=np.bool_),
        event_happened=event_happened[None],
        event_positions=event_positions[None],
        event_colours=event_colours[None],
        event_types=event_types[None],
        mission_present=np.asarray([True]),
        mission_position=np.asarray([[2, 2]], dtype=np.int32),
        mission_colour=np.asarray([0], dtype=np.int32),
        mission_event_type=np.asarray([0], dtype=np.int32),
        episode_returns=np.asarray([1.0], dtype=np.float32),
        episode_lengths=np.asarray([5], dtype=np.int32),
        episode_done=np.asarray([True]),
    )


def _symbolic_state(timestep: int, *, two_balls: bool) -> np.ndarray:
    frame = np.ones((4, 4, 3), dtype=np.int32)
    frame[0, :, 0] = 2
    frame[-1, :, 0] = 2
    frame[:, 0, 0] = 2
    frame[:, -1, 0] = 2
    if timestep < 3:
        frame[1, 2] = np.asarray([5, 2, 0], dtype=np.int32)
    door_state = 2 if timestep <= 3 else 0
    frame[1, 3] = np.asarray([4, 3, door_state], dtype=np.int32)
    ball_position = (2, 1) if timestep == 0 else (2, 2)
    frame[ball_position] = np.asarray([6, 4, 0], dtype=np.int32)
    if two_balls:
        frame[1, 1] = np.asarray([6, 4, 0], dtype=np.int32)
    return frame


def _set_event(
    happened: np.ndarray,
    positions: np.ndarray,
    colours: np.ndarray,
    event_types: np.ndarray,
    *,
    timestep: int,
    slot: int,
    position: tuple[int, int],
    colour: int,
    event_type: int,
) -> None:
    happened[timestep, slot] = True
    positions[timestep, slot] = position
    colours[timestep, slot] = colour
    event_types[timestep, slot] = event_type
