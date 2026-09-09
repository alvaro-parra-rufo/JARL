"""Typed contracts for Navix rollout capture and deterministic summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Self, cast

import jax
import numpy as np

VideoViewMode = Literal["full", "first_person"]
"""Supported RGB views for optional Navix rollout rendering."""

EventSource = Literal["navix", "derived"]
"""Origin of a normalized rollout event."""

EntityIdentity = Literal["unique", "ambiguous"]
"""Confidence in a dynamic entity correspondence between states."""

Position = tuple[int, int]
"""Grid position represented as ``(row, column)``."""

TraceArray = jax.Array | np.ndarray
"""Device or host array contained in a rollout trace."""

NAVIX_EVENT_TYPE_CODES: dict[str, int] = {
    "none": 0,
    "reach": 1,
    "hit": 2,
    "fall": 3,
    "pickup": 4,
    "open": 5,
    "unlock": 6,
}
"""Integer encoding of Navix ``EventType`` strings stored in traces."""

NAVIX_EVENT_SLOT_SPECS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("goal_reached", (("goal", "reach"),)),
    ("ball_hit", (("ball", "hit"),)),
    ("wall_hit", (("wall", "hit"), ("grid", "hit"))),
    ("lava_fall", (("lava", "fall"),)),
    ("key_pickup", (("key", "pickup"),)),
    ("door_opening", (("door", "open"),)),
    ("door_unlock", (("door", "unlock"),)),
    ("ball_pickup", (("ball", "pickup"),)),
    ("box_pickup", (("box", "pickup"),)),
)
"""Stable projection groups: slot name and Navix ``(entity, event_type)`` keys.

``wall_hit`` ORs a ``Wall`` entity hit with a grid-boundary hit. Maps that
never construct a key omit that leaf; capture writes ``happened=False``.
"""

NAVIX_EVENT_SLOT_NAMES: tuple[str, ...] = tuple(name for name, _keys in NAVIX_EVENT_SLOT_SPECS)
"""Stable field order used to project ``EventsManager`` arrays."""

__all__ = [
    "NAVIX_EVENT_SLOT_NAMES",
    "NAVIX_EVENT_SLOT_SPECS",
    "NAVIX_EVENT_TYPE_CODES",
    "ActionSegment",
    "ActionSummary",
    "DerivedEvent",
    "DynamicEntitySummary",
    "EntityCount",
    "EntityIdentity",
    "EntityTransition",
    "EventCount",
    "EventSource",
    "EventSummary",
    "InitialStateSummary",
    "NamedCount",
    "NavixCaptureProfile",
    "NavixRolloutIdentity",
    "NavixRolloutSummary",
    "NavixSummaryLimits",
    "NavixTraceArrays",
    "OutcomeSummary",
    "Position",
    "StepInterval",
    "SummarySectionCount",
    "TraceArray",
    "TrajectorySummary",
    "VideoViewMode",
    "VisibilitySummary",
]


@dataclass(frozen=True, slots=True)
class NavixCaptureProfile:
    """Static selection of arrays produced by a compiled Navix rollout."""

    capture_symbolic: bool
    capture_policy_observation: bool
    capture_player: bool
    capture_events: bool
    rgb_view_mode: VideoViewMode | None

    @classmethod
    def analysis(
        cls,
        *,
        record_video: bool = False,
        view_mode: VideoViewMode = "full",
    ) -> NavixCaptureProfile:
        """Return the rich symbolic profile used for rollout analysis."""
        _validate_view_mode(view_mode)
        return cls(
            capture_symbolic=True,
            capture_policy_observation=True,
            capture_player=True,
            capture_events=True,
            rgb_view_mode=view_mode if record_video else None,
        )

    @classmethod
    def video(cls, view_mode: VideoViewMode = "full") -> NavixCaptureProfile:
        """Return the reduced RGB-only profile used by policy videos."""
        _validate_view_mode(view_mode)
        return cls(
            capture_symbolic=False,
            capture_policy_observation=False,
            capture_player=False,
            capture_events=False,
            rgb_view_mode=view_mode,
        )


class NavixTraceArrays(NamedTuple):
    """Fixed-shape batched arrays returned by ``NavixTelemetryRollout``."""

    full_symbolic: TraceArray | None
    first_person_symbolic: TraceArray | None
    policy_observations: TraceArray | None
    player_positions: TraceArray | None
    player_directions: TraceArray | None
    player_pockets: TraceArray | None
    rgb_frames: TraceArray | None
    actions: TraceArray
    rewards: TraceArray
    step_types: TraceArray
    dones: TraceArray
    active_mask: TraceArray
    event_happened: TraceArray | None
    event_positions: TraceArray | None
    event_colours: TraceArray | None
    event_types: TraceArray | None
    mission_present: TraceArray
    mission_position: TraceArray
    mission_colour: TraceArray
    mission_event_type: TraceArray
    episode_returns: TraceArray
    episode_lengths: TraceArray
    episode_done: TraceArray

    def to_host(self) -> Self:
        """Transfer the complete trace pytree to host memory in one operation."""
        return cast(Self, jax.device_get(self))


@dataclass(frozen=True, slots=True)
class NavixSummaryLimits:
    """Maximum detailed entries retained in one compact rollout summary."""

    max_action_segments: int = 10000
    max_events: int = 10000
    max_visibility_intervals: int = 10000
    max_entity_transitions: int = 10000
    max_path_positions: int = 10000

    def __post_init__(self) -> None:
        """Validate that every summary limit is positive."""
        for field_name in (
            "max_action_segments",
            "max_events",
            "max_visibility_intervals",
            "max_entity_transitions",
            "max_path_positions",
        ):
            value = getattr(self, field_name)
            if value <= 0:
                msg = f"{field_name} must be > 0, got {value}."
                raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class StepInterval:
    """Inclusive interval of rollout state timesteps."""

    start: int
    end: int

    def __post_init__(self) -> None:
        """Validate inclusive interval ordering."""
        if self.start < 0:
            msg = f"Interval start must be >= 0, got {self.start}."
            raise ValueError(msg)
        if self.end < self.start:
            msg = f"Interval end ({self.end}) must be >= start ({self.start})."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class NamedCount:
    """Stable name and exact occurrence count."""

    name: str
    count: int


@dataclass(frozen=True, slots=True)
class SummarySectionCount:
    """Total and returned entries for one bounded summary section."""

    total: int
    returned: int

    @property
    def truncated(self) -> bool:
        """Return whether detailed entries were omitted."""
        return self.returned < self.total


@dataclass(frozen=True, slots=True)
class ActionSegment:
    """Consecutive run of one action over transition timesteps."""

    start: int
    end: int
    action_index: int
    action_name: str


@dataclass(frozen=True, slots=True)
class DerivedEvent:
    """Physical event normalized from Navix payloads or state differences."""

    timestep: int
    kind: str
    source: EventSource
    position: Position | None = None
    colour: int | None = None
    event_type: int | None = None
    slot: str | None = None
    ambiguous: bool = False


@dataclass(frozen=True, slots=True)
class EventCount:
    """Exact event count grouped by normalized kind and source."""

    kind: str
    source: EventSource
    count: int


@dataclass(frozen=True, slots=True)
class EntityTransition:
    """Observed occupancy change for a dynamic symbolic entity."""

    timestep: int
    entity_tag: int
    entity_name: str
    colour: int
    state: int
    from_positions: tuple[Position, ...]
    to_positions: tuple[Position, ...]
    identity: EntityIdentity


@dataclass(frozen=True, slots=True)
class EntityCount:
    """Initial count of one symbolic entity descriptor."""

    entity_tag: int
    entity_name: str
    colour: int
    state: int
    count: int


@dataclass(frozen=True, slots=True)
class VisibilitySummary:
    """Visibility intervals and totals for one symbolic entity descriptor."""

    entity_tag: int
    entity_name: str
    colour: int
    state: int
    ever_seen: bool
    first_seen: int | None
    last_seen: int | None
    visible_steps: int
    interval_count: SummarySectionCount
    intervals: tuple[StepInterval, ...]
    formatted_intervals: str


@dataclass(frozen=True, slots=True)
class InitialStateSummary:
    """Compact description of the rollout's initial Navix state."""

    map_shape: tuple[int, int, int]
    map_hash: str
    player_position: Position
    player_direction: int
    player_pocket: int
    mission: DerivedEvent | None
    entities: tuple[EntityCount, ...]


@dataclass(frozen=True, slots=True)
class OutcomeSummary:
    """Episode return and final state classification."""

    return_total: float
    length: int
    terminated: bool
    truncated: bool
    success: bool | None
    reason: str


@dataclass(frozen=True, slots=True)
class ActionSummary:
    """Action counts and bounded run-length timeline."""

    counts: tuple[NamedCount, ...]
    segment_count: SummarySectionCount
    segments: tuple[ActionSegment, ...]


@dataclass(frozen=True, slots=True)
class EventSummary:
    """Event counts and bounded normalized timeline."""

    counts: tuple[EventCount, ...]
    timeline_count: SummarySectionCount
    timeline: tuple[DerivedEvent, ...]


@dataclass(frozen=True, slots=True)
class TrajectorySummary:
    """Compact agent path and movement statistics."""

    start: Position
    end: Position
    unique_cells: int
    manhattan_distance: int
    rotations: int
    blocked_actions: int
    path_count: SummarySectionCount
    path: tuple[Position, ...]


@dataclass(frozen=True, slots=True)
class DynamicEntitySummary:
    """Bounded dynamic-entity transition evidence."""

    transition_count: SummarySectionCount
    transitions: tuple[EntityTransition, ...]


@dataclass(frozen=True, slots=True)
class NavixRolloutIdentity:
    """Identity fields known while summarizing one rollout episode."""

    env_id: str
    seed: int | None = None
    rollout_id: str | None = None
    node_id: str | None = None
    checkpoint_step: int | None = None
    global_step: int | None = None
    algorithm_name: str | None = None
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class NavixRolloutSummary:
    """Deterministic compact summary of one captured Navix episode."""

    identity: NavixRolloutIdentity
    outcome: OutcomeSummary
    initial_state: InitialStateSummary
    actions: ActionSummary
    events: EventSummary
    visibility: tuple[VisibilitySummary, ...]
    trajectory: TrajectorySummary
    dynamic_entities: DynamicEntitySummary
    artifact_paths: tuple[str, ...] = ()

    def as_text(
        self,
        *,
        compact: bool = False,
        include_timeline: bool = True,
        include_path: bool = False,
        include_artifacts: bool | None = None,
    ) -> str:
        """Render the summary as plain text for humans or LLM consumers.

        Args:
            compact: When ``True``, use a terse sectioned layout suited for
                tools and LLM prompts. When ``False``, use a readable report.
            include_timeline: Include bounded action segments and event entries.
            include_path: Include bounded trajectory coordinates.
            include_artifacts: Include persisted artifact relative paths. When
                ``None``, artifacts are shown in human mode and omitted in
                compact mode.

        Returns:
            Multi-line plain-text rendering of this summary.
        """
        from jarl.envs.navix.telemetry.display import format_rollout_summary

        return format_rollout_summary(
            self,
            compact=compact,
            include_timeline=include_timeline,
            include_path=include_path,
            include_artifacts=include_artifacts,
        )

    def show(
        self,
        *,
        compact: bool = False,
        include_timeline: bool = True,
        include_path: bool = False,
        include_artifacts: bool | None = None,
    ) -> None:
        """Display the summary in Jupyter when available, otherwise write stdout."""
        import sys

        text = self.as_text(
            compact=compact,
            include_timeline=include_timeline,
            include_path=include_path,
            include_artifacts=include_artifacts,
        )
        try:
            ipython = get_ipython()  # type: ignore[name-defined]
        except NameError:
            sys.stdout.write(f"{text}\n")
            return
        if ipython is not None and hasattr(ipython, "run_line_magic"):
            from IPython.display import Markdown, display

            display(Markdown(f"```\n{text}\n```"))
            return
        sys.stdout.write(f"{text}\n")

    def __str__(self) -> str:
        """Return the human-readable text rendering."""
        return self.as_text()


def _validate_view_mode(view_mode: str) -> None:
    if view_mode in {"full", "first_person"}:
        return
    msg = f"Unsupported Navix video view mode: {view_mode!r}."
    raise ValueError(msg)
