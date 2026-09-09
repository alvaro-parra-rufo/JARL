"""Anti-leak tests for checkpoint / recovery / metrics semantic boundaries."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from jarl.agentic.langgraph.graphs.subagents.metrics_analysis import (
    MetricsAnalysisOutput,
    parse_metrics_analysis_payload,
    validate_evidence_subset,
)
from jarl.agentic.tools.registry import REGISTRY
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import CHECKPOINT_BEST_RETURN_METRIC
from jarl.operations.graph.checkpoints import CheckpointsRequest, checkpoints
from jarl.operations.subagent.metrics_preprocess import build_metrics_features
from jarl.operations.train.recovery_status import RecoveryStatusRequest, recovery_status
from jarl.training.config import RLRunConfig

_REMOVED_MODULES = (
    "jarl.agentic.tools.subagent.summarize_metrics",
    "jarl.agentic.tools.subagent.summarize_checkpoint_events",
    "jarl.agentic.langgraph.graphs.subagents.summarize_metrics",
    "jarl.agentic.langgraph.graphs.subagents.summarize_checkpoint_events",
)

_FORBIDDEN_PREPROCESS_KEYS = (
    "trend",
    "is_spike",
    "collapse_score",
    "anomalies",
    "labels",
    "recommended_action",
)


@pytest.fixture()
def prepared_root(prepared_experiment: Path) -> ExperimentGraph[RLRunConfig]:
    """Reload the prepared experiment as an ``ExperimentGraph``."""
    return ExperimentGraph.from_directory(prepared_experiment, config_cls=RLRunConfig)


class TestSummarizeHardCut:
    def test_registry_has_no_summarize_tools(self) -> None:
        assert not any(name.startswith("subagent_summarize_") for name in REGISTRY.names())
        assert "subagent_metrics_analysis" in REGISTRY.names()
        assert "train_recovery_status" in REGISTRY.names()
        assert "graph_checkpoints" in REGISTRY.names()

    @pytest.mark.parametrize("module_path", _REMOVED_MODULES)
    def test_summarize_modules_are_gone(self, module_path: str) -> None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_path)


class TestCheckpointsNoForeignSemantics:
    def test_overview_excludes_recovery_and_metrics_labels(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        workspace = prepared_root.current_node
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        workspace.save_checkpoint(
            8,
            {"value": 8},
            metrics={CHECKPOINT_BEST_RETURN_METRIC: 0.5},
            force=True,
        )
        prepared_root.save()

        compact = checkpoints(prepared_root, CheckpointsRequest()).to_compact_dict()

        assert "latest" in compact
        assert "recommended_action" not in compact
        assert "can_resume" not in compact
        assert "resume_hint" not in compact
        assert "labels" not in compact
        assert "evidence" not in compact
        assert "summary" not in compact


class TestRecoveryNoRankingOrNarrative:
    def test_recovery_excludes_aliases_and_curve_labels(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        workspace = prepared_root.current_node
        workspace.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        try:
            with workspace:
                workspace.save_checkpoint(3, {"value": 3}, force=True)
                raise RuntimeError("boundary boom")
        except RuntimeError:
            pass
        prepared_root.save()

        compact = recovery_status(prepared_root, RecoveryStatusRequest()).to_compact_dict()

        assert compact["recommended_action"] == "train_resume"
        assert "resume_hint" not in compact
        assert "latest" not in compact
        assert "best" not in compact
        assert "final" not in compact
        assert "candidates" not in compact
        assert "labels" not in compact
        assert "evidence" not in compact


class TestMetricsPreprocessObjectiveOnly:
    def test_features_exclude_heuristic_conclusions(self) -> None:
        payload = build_metrics_features(
            "node",
            {"eval/episode_return": [(i, float(i)) for i in range(9)]},
        )

        joined = " ".join(payload.features)
        for forbidden in _FORBIDDEN_PREPROCESS_KEYS:
            assert forbidden not in joined
            assert forbidden not in payload.features

    def test_evidence_must_be_feature_subset(self) -> None:
        output = MetricsAnalysisOutput(
            labels=["degrading"],
            summary="Late window is worse.",
            evidence=["eval_return.window_final.mean", "not_a_real_feature"],
        )

        with pytest.raises(ValueError, match="unknown feature ids"):
            validate_evidence_subset(
                output,
                {"eval_return.window_final.mean", "eval_return.window_initial.mean"},
            )

    def test_stalled_label_is_coerced_to_plateau(self) -> None:
        output = parse_metrics_analysis_payload(
            {
                "labels": ["stalled"],
                "summary": "No reward observed.",
                "evidence": ["eval_return.last.value"],
            }
        )

        assert output.labels == ["plateau"]
