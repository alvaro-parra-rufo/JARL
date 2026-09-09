"""Human and LLM-oriented text rendering for rollout summaries."""

from __future__ import annotations

from collections.abc import Iterable

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
    OutcomeSummary,
    SummarySectionCount,
    TrajectorySummary,
    VisibilitySummary,
)

__all__ = ["format_rollout_summary"]


def format_rollout_summary(
    summary: NavixRolloutSummary,
    *,
    compact: bool = False,
    include_timeline: bool = True,
    include_path: bool = False,
    include_artifacts: bool | None = None,
) -> str:
    """Render one rollout summary as plain text.

    Args:
        summary: Deterministic rollout summary to format.
        compact: When ``True``, emit a terse LLM-oriented layout with stable
            section headers. When ``False``, emit a human-readable report.
        include_timeline: Include bounded action segments and event timelines.
        include_path: Include the bounded agent path coordinates.
        include_artifacts: Include persisted artifact relative paths. When
            ``None``, artifacts are shown in human mode and omitted in compact
            mode.

    Returns:
        Multi-line plain-text rendering of ``summary``.
    """
    if include_artifacts is None:
        include_artifacts = not compact
    if compact:
        sections = _compact_sections(
            summary,
            include_timeline=include_timeline,
            include_path=include_path,
            include_artifacts=include_artifacts,
        )
    else:
        sections = _human_sections(
            summary,
            include_timeline=include_timeline,
            include_path=include_path,
            include_artifacts=include_artifacts,
        )
    return "\n".join(line for section in sections for line in section)


def _human_sections(
    summary: NavixRolloutSummary,
    *,
    include_timeline: bool,
    include_path: bool,
    include_artifacts: bool,
) -> list[list[str]]:
    return [
        _human_header(summary.identity),
        _human_outcome(summary.outcome),
        _human_initial_state(summary.initial_state),
        _human_actions(summary.actions, include_timeline=include_timeline),
        _human_events(summary.events, include_timeline=include_timeline),
        _human_visibility(summary.visibility),
        _human_trajectory(summary.trajectory, include_path=include_path),
        _human_dynamic_entities(summary.dynamic_entities),
        *_human_artifacts(summary.artifact_paths, include_artifacts=include_artifacts),
    ]


def _compact_sections(
    summary: NavixRolloutSummary,
    *,
    include_timeline: bool,
    include_path: bool,
    include_artifacts: bool,
) -> list[list[str]]:
    return [
        _compact_identity(summary.identity),
        _compact_outcome(summary.outcome),
        _compact_initial_state(summary.initial_state),
        _compact_actions(summary.actions, include_timeline=include_timeline),
        _compact_events(summary.events, include_timeline=include_timeline),
        _compact_visibility(summary.visibility),
        _compact_trajectory(summary.trajectory, include_path=include_path),
        _compact_dynamic_entities(summary.dynamic_entities),
        *_compact_artifacts(summary.artifact_paths, include_artifacts=include_artifacts),
    ]


def _human_header(identity: NavixRolloutIdentity) -> list[str]:
    lines = [
        "Navix rollout summary",
        "=====================",
        f"Environment:  {identity.env_id}",
    ]
    if identity.node_id is not None:
        lines.append(f"Node:         {identity.node_id}")
    if identity.checkpoint_step is not None:
        lines.append(f"Checkpoint:   {identity.checkpoint_step}")
    if identity.seed is not None:
        lines.append(f"Seed:         {identity.seed}")
    if identity.rollout_id is not None:
        lines.append(f"Rollout id:   {identity.rollout_id}")
    if identity.algorithm_name is not None:
        lines.append(f"Algorithm:    {identity.algorithm_name}")
    if identity.global_step is not None:
        lines.append(f"Global step:  {identity.global_step}")
    return lines


def _human_outcome(outcome: OutcomeSummary) -> list[str]:
    return [
        "",
        "Outcome",
        "-------",
        f"Return:       {outcome.return_total:.3f}",
        f"Length:       {outcome.length}",
        f"Success:      {_format_optional_bool(outcome.success)}",
        f"Reason:       {outcome.reason}",
        f"Terminated:   {_yes_no(outcome.terminated)}",
        f"Truncated:    {_yes_no(outcome.truncated)}",
    ]


def _human_initial_state(initial_state: InitialStateSummary) -> list[str]:
    lines = [
        "",
        "Initial state",
        "-------------",
        f"Map shape:    {initial_state.map_shape}",
        f"Map hash:     {initial_state.map_hash}",
        f"Start:        {initial_state.player_position}",
        f"Direction:    {initial_state.player_direction}",
        f"Pocket:       {initial_state.player_pocket}",
    ]
    if initial_state.mission is not None:
        mission = initial_state.mission
        lines.append(
            f"Mission:      {mission.kind} @ {mission.position} (colour={mission.colour}, type={mission.event_type})"
        )
    if initial_state.entities:
        lines.append("Entities:")
        lines.extend(_entity_inventory_lines(initial_state.entities))
    return lines


def _human_actions(actions: ActionSummary, *, include_timeline: bool) -> list[str]:
    lines = [
        "",
        "Actions",
        "-------",
        *_named_count_lines(actions.counts, prefix="  "),
        f"Segments:     {_section_count_label(actions.segment_count)}",
    ]
    if include_timeline and actions.segments:
        lines.append("Timeline:")
        lines.extend(_action_segment_lines(actions.segments, prefix="  "))
    return lines


def _human_events(events: EventSummary, *, include_timeline: bool) -> list[str]:
    lines = [
        "",
        "Events",
        "------",
        *_event_count_lines(events.counts, prefix="  "),
        f"Timeline:     {_section_count_label(events.timeline_count)}",
    ]
    if include_timeline and events.timeline:
        lines.append("Entries:")
        lines.extend(_event_timeline_lines(events.timeline, prefix="  "))
    return lines


def _human_visibility(visibility: tuple[VisibilitySummary, ...]) -> list[str]:
    visible = [item for item in visibility if item.ever_seen]
    lines = ["", "Visibility", "----------"]
    if visible:
        lines.extend(
            (
                f"  - {item.entity_name}: steps={item.visible_steps}, "
                f"intervals={item.formatted_intervals or '—'} "
                f"({_section_count_label(item.interval_count)})"
            )
            for item in visible
        )
    else:
        lines.append("  (none)")
    return lines


def _human_trajectory(trajectory: TrajectorySummary, *, include_path: bool) -> list[str]:
    lines = [
        "",
        "Trajectory",
        "----------",
        f"Start / end:  {trajectory.start} -> {trajectory.end}",
        f"Unique cells: {trajectory.unique_cells}",
        f"Manhattan:    {trajectory.manhattan_distance}",
        f"Rotations:    {trajectory.rotations}",
        f"Blocked:      {trajectory.blocked_actions}",
        f"Path:         {_section_count_label(trajectory.path_count)}",
    ]
    if include_path and trajectory.path:
        coordinates = " -> ".join(str(position) for position in trajectory.path)
        lines.append(f"Coordinates:  {coordinates}")
    return lines


def _human_dynamic_entities(dynamic_entities: DynamicEntitySummary) -> list[str]:
    lines = [
        "",
        "Dynamic entities",
        "----------------",
        f"Transitions:  {_section_count_label(dynamic_entities.transition_count)}",
    ]
    if dynamic_entities.transitions:
        lines.extend(
            (
                f"  - t={transition.timestep} {transition.entity_name} "
                f"{transition.from_positions} -> {transition.to_positions} "
                f"({transition.identity})"
            )
            for transition in dynamic_entities.transitions
        )
    return lines


def _human_artifacts(
    artifact_paths: tuple[str, ...],
    *,
    include_artifacts: bool,
) -> list[list[str]]:
    if not include_artifacts or not artifact_paths:
        return []
    return [["", "Artifacts", "---------", *(f"  - {path}" for path in artifact_paths)]]


def _compact_identity(identity: NavixRolloutIdentity) -> list[str]:
    lines = ["[identity]", f"env_id={identity.env_id}"]
    if identity.node_id is not None:
        lines.append(f"node_id={identity.node_id}")
    if identity.checkpoint_step is not None:
        lines.append(f"checkpoint_step={identity.checkpoint_step}")
    if identity.seed is not None:
        lines.append(f"seed={identity.seed}")
    if identity.rollout_id is not None:
        lines.append(f"rollout_id={identity.rollout_id}")
    if identity.algorithm_name is not None:
        lines.append(f"algorithm_name={identity.algorithm_name}")
    if identity.global_step is not None:
        lines.append(f"global_step={identity.global_step}")
    return lines


def _compact_outcome(outcome: OutcomeSummary) -> list[str]:
    return [
        "",
        "[outcome]",
        f"return_total={outcome.return_total:.3f}",
        f"length={outcome.length}",
        f"success={_format_optional_bool(outcome.success, compact=True)}",
        f"reason={outcome.reason}",
        f"terminated={outcome.terminated}",
        f"truncated={outcome.truncated}",
    ]


def _compact_initial_state(initial_state: InitialStateSummary) -> list[str]:
    lines = [
        "",
        "[initial_state]",
        f"map_shape={initial_state.map_shape}",
        f"map_hash={initial_state.map_hash}",
        f"player_position={initial_state.player_position}",
        f"player_direction={initial_state.player_direction}",
        f"player_pocket={initial_state.player_pocket}",
    ]
    if initial_state.mission is not None:
        mission = initial_state.mission
        lines.append(f"mission={mission.kind}@{mission.position}:colour={mission.colour}:type={mission.event_type}")
    if initial_state.entities:
        lines.append(
            "entities=" + ",".join(f"{entity.entity_name}:{entity.count}" for entity in initial_state.entities)
        )
    return lines


def _compact_actions(actions: ActionSummary, *, include_timeline: bool) -> list[str]:
    lines = [
        "",
        "[actions]",
        "counts=" + _join_counts((item.name, item.count) for item in actions.counts),
        f"segments={_section_count_compact(actions.segment_count)}",
    ]
    if include_timeline and actions.segments:
        lines.append("timeline=" + ";".join(_action_segment_compact(segment) for segment in actions.segments))
    return lines


def _compact_events(events: EventSummary, *, include_timeline: bool) -> list[str]:
    lines = [
        "",
        "[events]",
        "counts=" + _join_counts((f"{item.kind}:{item.source}", item.count) for item in events.counts),
        f"timeline={_section_count_compact(events.timeline_count)}",
    ]
    if include_timeline and events.timeline:
        lines.append("entries=" + ";".join(_event_timeline_compact(event) for event in events.timeline))
    return lines


def _compact_visibility(visibility: tuple[VisibilitySummary, ...]) -> list[str]:
    visible = [item for item in visibility if item.ever_seen]
    lines = ["", "[visibility]"]
    if visible:
        lines.extend(_visibility_compact(item) for item in visible)
    else:
        lines.append("none")
    return lines


def _compact_trajectory(trajectory: TrajectorySummary, *, include_path: bool) -> list[str]:
    lines = [
        "",
        "[trajectory]",
        f"start={trajectory.start}",
        f"end={trajectory.end}",
        f"unique_cells={trajectory.unique_cells}",
        f"manhattan_distance={trajectory.manhattan_distance}",
        f"rotations={trajectory.rotations}",
        f"blocked_actions={trajectory.blocked_actions}",
        f"path={_section_count_compact(trajectory.path_count)}",
    ]
    if include_path and trajectory.path:
        lines.append("coordinates=" + "->".join(str(position) for position in trajectory.path))
    return lines


def _compact_dynamic_entities(dynamic_entities: DynamicEntitySummary) -> list[str]:
    if dynamic_entities.transition_count.total == 0 and not dynamic_entities.transitions:
        return []
    lines = [
        "",
        "[dynamic_entities]",
        f"transitions={_section_count_compact(dynamic_entities.transition_count)}",
    ]
    if dynamic_entities.transitions:
        lines.append(
            "entries="
            + ";".join(
                f"t{transition.timestep}:{transition.entity_name}:"
                f"{transition.from_positions}->{transition.to_positions}:"
                f"{transition.identity}"
                for transition in dynamic_entities.transitions
            )
        )
    return lines


def _compact_artifacts(
    artifact_paths: tuple[str, ...],
    *,
    include_artifacts: bool,
) -> list[list[str]]:
    if not include_artifacts or not artifact_paths:
        return []
    return [["", "[artifacts]", *artifact_paths]]


def _entity_inventory_lines(entities: tuple[EntityCount, ...]) -> list[str]:
    return [
        (
            f"  - {entity.entity_name} "
            f"(tag={entity.entity_tag}, colour={entity.colour}, "
            f"state={entity.state}): {entity.count}"
        )
        for entity in entities
    ]


def _format_optional_bool(value: bool | None, *, compact: bool = False) -> str:
    if value is None:
        return "unknown"
    if compact:
        return str(value).lower()
    return "yes" if value else "no"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _section_count_label(count: SummarySectionCount) -> str:
    if count.truncated:
        return f"{count.returned}/{count.total} (truncated)"
    return str(count.total)


def _section_count_compact(count: SummarySectionCount) -> str:
    if count.truncated:
        return f"{count.returned}/{count.total}|truncated"
    return str(count.total)


def _named_count_lines(counts: tuple[NamedCount, ...], *, prefix: str) -> list[str]:
    if not counts:
        return [f"{prefix}(none)"]
    return [f"{prefix}{item.name}: {item.count}" for item in counts]


def _event_count_lines(counts: tuple[EventCount, ...], *, prefix: str) -> list[str]:
    if not counts:
        return [f"{prefix}(none)"]
    return [f"{prefix}{item.kind} ({item.source}): {item.count}" for item in counts]


def _action_segment_lines(segments: tuple[ActionSegment, ...], *, prefix: str) -> list[str]:
    return [
        f"{prefix}t{segment.start}-{segment.end}: {segment.action_name} ({segment.action_index})"
        for segment in segments
    ]


def _event_timeline_lines(events: tuple[DerivedEvent, ...], *, prefix: str) -> list[str]:
    return [
        f"{prefix}t{event.timestep}: {event.kind} [{event.source}]" + (" ambiguous" if event.ambiguous else "")
        for event in events
    ]


def _action_segment_compact(segment: ActionSegment) -> str:
    return f"t{segment.start}-{segment.end}:{segment.action_name}"


def _event_timeline_compact(event: DerivedEvent) -> str:
    suffix = ":ambiguous" if event.ambiguous else ""
    return f"t{event.timestep}:{event.kind}:{event.source}{suffix}"


def _visibility_compact(item: VisibilitySummary) -> str:
    return (
        f"{item.entity_name}:seen={item.ever_seen}:steps={item.visible_steps}:"
        f"intervals={item.formatted_intervals or '-'}:"
        f"count={_section_count_compact(item.interval_count)}"
    )


def _join_counts(pairs: Iterable[tuple[str, int]]) -> str:
    rendered = [f"{name}={count}" for name, count in pairs]
    return ",".join(rendered) if rendered else "-"
