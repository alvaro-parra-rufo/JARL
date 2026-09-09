"""Tests for objective metrics preprocess and metrics_analysis operation."""

from __future__ import annotations

import pytest

from jarl.agentic.langgraph.graphs.subagents.metrics_analysis import (
    MetricsAnalysisOutput,
    validate_evidence_subset,
)
from jarl.experiments.graph import ExperimentGraph
from jarl.operations.subagent.metrics_analysis import MetricsAnalysisRequest, metrics_analysis
from jarl.operations.subagent.metrics_preprocess import build_metrics_features
from jarl.training.config import RLRunConfig


class TestMetricsPreprocess:
    def test_builds_window_and_delta_features(self) -> None:
        payload = build_metrics_features(
            "node_a",
            {"eval/episode_return": [(i, float(i)) for i in range(9)]},
        )

        assert "eval_return.window_initial.mean" in payload.features
        assert "eval_return.window_final.mean" in payload.features
        assert "eval_return.delta_abs" in payload.features
        assert payload.features["eval_return.delta_abs"] == 8.0
        assert payload.features["eval_return.p25"] == pytest.approx(2.0)
        assert payload.features["eval_return.p50"] == pytest.approx(4.0)
        assert payload.features["eval_return.p75"] == pytest.approx(6.0)
        assert payload.features["eval_return.p90"] == pytest.approx(7.2)
        assert "eval_return.window_final.p50" in payload.features
        assert "trend" not in payload.features
        assert "is_spike" not in payload.features
        assert set(payload.feature_ids) == set(payload.features)

    def test_empty_series_keeps_count_only(self) -> None:
        payload = build_metrics_features("node_a", {"eval/episode_return": []})

        assert payload.features == {"eval_return.n_points": 0}


class TestMetricsAnalysisOperation:
    def test_operation_reads_logged_metrics(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        workspace = prepared_root.current_node
        for step in range(6):
            workspace.log_scalar(step, "eval/episode_return", float(step))
        prepared_root.save()

        response = metrics_analysis(
            prepared_root,
            MetricsAnalysisRequest(node_id=workspace.id, focus="eval"),
        )

        assert response.node_id == workspace.id
        assert response.focus == "eval"
        assert response.features["eval_return.last.value"] == 5.0
        assert "eval_return.window_final.mean" in response.feature_ids


class TestEvidenceValidation:
    def test_rejects_unknown_evidence_ids(self) -> None:
        output = MetricsAnalysisOutput(
            labels=["improving"],
            summary="Looks better.",
            evidence=["eval_return.window_final.mean", "invented.feature"],
        )

        with pytest.raises(ValueError, match="unknown feature ids"):
            validate_evidence_subset(output, {"eval_return.window_final.mean"})
