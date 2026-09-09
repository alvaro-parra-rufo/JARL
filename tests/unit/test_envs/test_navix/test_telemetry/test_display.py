"""Tests for rollout summary text rendering."""

from __future__ import annotations

from dataclasses import replace

from jarl.envs.navix.telemetry import (
    DynamicEntitySummary,
    NavixRolloutIdentity,
    NavixRolloutSummary,
    NavixSummaryLimits,
    SummarySectionCount,
    format_rollout_summary,
    summarize_navix_trace,
)
from tests.unit.test_envs.test_navix.test_telemetry.test_postprocess import _trace

_ACTION_NAMES = (
    "rotate_ccw",
    "rotate_cw",
    "forward",
    "pickup",
    "drop",
    "toggle",
    "done",
)


class TestRolloutSummaryDisplay:
    """Human and compact text renderings stay stable and bounded."""

    def test_human_text_includes_readable_sections(self) -> None:
        summary = _summary()

        text = summary.as_text()

        assert "Navix rollout summary" in text
        assert "Outcome" in text
        assert "Return:       1.000" in text
        assert "goal_reached" in text
        assert "Visibility" in text
        assert str(summary) == text

    def test_compact_text_uses_stable_llm_sections(self) -> None:
        summary = _summary()

        text = summary.as_text(compact=True)

        assert "[identity]" in text
        assert "env_id=Navix-Test-v0" in text
        assert "[outcome]" in text
        assert "success=true" in text
        assert "[events]" in text
        assert "goal_reached:navix=1" in text
        assert "Navix rollout summary" not in text

    def test_truncation_is_visible_in_both_formats(self) -> None:
        summary = summarize_navix_trace(
            _trace(),
            identity=NavixRolloutIdentity(env_id="Navix-Test-v0", seed=7),
            action_names=_ACTION_NAMES,
            limits=NavixSummaryLimits(max_action_segments=1, max_events=1),
        )

        human = format_rollout_summary(summary)
        compact = format_rollout_summary(summary, compact=True)

        assert "1/4 (truncated)" in human
        assert "segments=1/4|truncated" in compact

    def test_include_path_adds_coordinates_only_when_requested(self) -> None:
        summary = _summary()

        without_path = summary.as_text(include_path=False)
        with_path = summary.as_text(include_path=True)

        assert "Coordinates:" not in without_path
        assert "Coordinates:" in with_path

        compact_with_path = summary.as_text(compact=True, include_path=True)
        assert "coordinates=" in compact_with_path

    def test_compact_omits_artifacts_by_default(self) -> None:
        summary = _summary_with_artifacts()

        compact = summary.as_text(compact=True)

        assert "[artifacts]" not in compact
        assert "rollout.json" not in compact

        human = summary.as_text()
        assert "Artifacts" in human
        assert "rollout.json" in human

        compact_with_artifacts = summary.as_text(compact=True, include_artifacts=True)
        assert "[artifacts]" in compact_with_artifacts

    def test_compact_omits_empty_dynamic_entities(self) -> None:
        summary = replace(
            _summary(),
            dynamic_entities=DynamicEntitySummary(
                transition_count=SummarySectionCount(total=0, returned=0),
                transitions=(),
            ),
        )

        compact = summary.as_text(compact=True)

        assert "[dynamic_entities]" not in compact


def _summary() -> NavixRolloutSummary:
    return summarize_navix_trace(
        _trace(),
        identity=NavixRolloutIdentity(
            env_id="Navix-Test-v0",
            seed=7,
            node_id="main_rollout_ab12cd34",
            checkpoint_step=64,
            rollout_id="rollout123",
            algorithm_name="ppo.full_jax.navix",
        ),
        action_names=_ACTION_NAMES,
        limits=NavixSummaryLimits(max_path_positions=3),
    )


def _summary_with_artifacts() -> NavixRolloutSummary:
    return replace(
        _summary(),
        artifact_paths=(
            "rollouts/166716f5c7b950c6d0fd5834/rollout.json",
            "rollouts/166716f5c7b950c6d0fd5834/trace.npz",
            "rollouts/166716f5c7b950c6d0fd5834/rollout.mp4",
        ),
    )
