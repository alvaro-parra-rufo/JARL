"""Tests for Runner Lab layout helpers."""

from __future__ import annotations

from pathlib import Path

from jarl.app.lib.layout import APP_PAGES, format_node_label, latest_scalar_metrics


class TestFormatNodeLabel:
    """Tests for node label formatting."""

    def test_format_node_label_joins_id_and_status(self) -> None:
        assert format_node_label("node_a", "completed") == "node_a · completed"


class TestAppPages:
    """Tests for canonical page paths."""

    def test_app_pages_include_runner_lab_modules(self) -> None:
        assert set(APP_PAGES) == {
            "inicio",
            "entrenar",
            "metricas",
            "arbol",
            "inferencias",
            "testing",
        }
        assert APP_PAGES["metricas"].endswith("4_Metricas.py")
        assert APP_PAGES["entrenar"].endswith("3_Entrenar.py")
        assert APP_PAGES["arbol"].endswith("5_Arbol.py")
        assert APP_PAGES["inferencias"].endswith("8_Inferencias.py")
        assert APP_PAGES["testing"].endswith("9_Testing.py")


class TestLatestScalarMetrics:
    """Tests for metrics.jsonl tail reads."""

    def test_latest_scalar_metrics_returns_empty_when_missing_file(self, tmp_path: Path) -> None:
        class _Workspace:
            metrics_jsonl_path = tmp_path / "missing.jsonl"

        assert latest_scalar_metrics(_Workspace()) == {}

    def test_latest_scalar_metrics_reads_last_row(self, tmp_path: Path) -> None:
        metrics_path = tmp_path / "metrics.jsonl"
        metrics_path.write_text(
            '{"step": 1, "name": "rollout/episode_return", "value": 0.1}\n'
            '{"step": 2, "name": "eval/episode_return", "value": 0.9}\n',
            encoding="utf-8",
        )

        class _Workspace:
            metrics_jsonl_path = metrics_path

        latest = latest_scalar_metrics(_Workspace())

        assert latest["rollout/episode_return"] == 0.1
        assert latest["eval/episode_return"] == 0.9
