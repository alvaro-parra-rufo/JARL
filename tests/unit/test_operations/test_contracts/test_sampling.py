"""Tests for operation contract helpers."""

from __future__ import annotations

from jarl.operations.contracts.sampling import downsample_indices


def test_downsample_indices_returns_all_when_under_limit() -> None:
    assert downsample_indices(4, 10) == [0, 1, 2, 3]


def test_downsample_indices_is_deterministic() -> None:
    first = downsample_indices(100, 8)
    second = downsample_indices(100, 8)

    assert first == second
    assert first[0] == 0
    assert first[-1] == 99
