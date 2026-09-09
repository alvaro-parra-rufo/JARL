"""Tests for metric key validation."""

from __future__ import annotations

import pytest

from jarl.agents.ppo.metric_schema import suggest_ppo_metric_names
from jarl.experiments.metrics import validate_requested_metric_keys


class TestSuggestPpoMetricNames:
    def test_maps_explained_variance_typo(self) -> None:
        assert suggest_ppo_metric_names("loss/explained_variance") == ("v_value/explained_variance",)

    def test_maps_charts_prefix_to_rollout(self) -> None:
        assert suggest_ppo_metric_names("charts/episode_return") == ("rollout/episode_return",)


class TestValidateRequestedMetricKeys:
    def test_accepts_known_keys(self) -> None:
        validate_requested_metric_keys(
            ("loss/entropy_loss", "v_value/explained_variance"),
            ["loss/entropy_loss"],
        )

    def test_rejects_unknown_keys_with_available_list(self) -> None:
        with pytest.raises(ValueError, match="Unknown metric keys: loss/explained_variance"):
            validate_requested_metric_keys(
                ("loss/entropy_loss", "v_value/explained_variance"),
                ["loss/explained_variance"],
            )

    def test_rejects_unknown_keys_with_alias_hint(self) -> None:
        with pytest.raises(ValueError, match=r"loss/explained_variance.*v_value/explained_variance"):
            validate_requested_metric_keys(
                ("v_value/explained_variance",),
                ["loss/explained_variance"],
            )

    def test_rejects_when_node_has_no_metrics(self) -> None:
        with pytest.raises(ValueError, match="No metrics are logged"):
            validate_requested_metric_keys((), ["eval/episode_return"])
