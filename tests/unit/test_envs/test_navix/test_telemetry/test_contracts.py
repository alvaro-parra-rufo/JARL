"""Tests for Navix telemetry contracts."""

from __future__ import annotations

import pytest

from jarl.envs.navix.telemetry import (
    NavixCaptureProfile,
    NavixSummaryLimits,
    StepInterval,
    format_step_intervals,
)


class TestNavixCaptureProfile:
    """Capture profiles expose static analysis and video selections."""

    def test_analysis_omits_rgb_by_default(self) -> None:
        profile = NavixCaptureProfile.analysis()

        assert profile.capture_symbolic is True
        assert profile.capture_policy_observation is True
        assert profile.capture_player is True
        assert profile.capture_events is True
        assert profile.rgb_view_mode is None

    def test_video_only_requests_rgb(self) -> None:
        profile = NavixCaptureProfile.video("first_person")

        assert profile.capture_symbolic is False
        assert profile.capture_policy_observation is False
        assert profile.capture_player is False
        assert profile.capture_events is False
        assert profile.rgb_view_mode == "first_person"

    def test_unknown_view_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="view mode"):
            NavixCaptureProfile.video("unknown")  # type: ignore[arg-type]


class TestSummaryContracts:
    """Summary bounds and interval rendering remain explicit."""

    def test_limits_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_events"):
            NavixSummaryLimits(max_events=0)

    def test_format_step_intervals_uses_ranges_and_commas(self) -> None:
        intervals = (
            StepInterval(start=4, end=20),
            StepInterval(start=25, end=30),
            StepInterval(start=40, end=40),
        )

        formatted = format_step_intervals(intervals)

        assert formatted == "4-20, 25-30, 40"
