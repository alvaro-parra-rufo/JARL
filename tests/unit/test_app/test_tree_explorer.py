"""Tests for tree explorer SVG and PNG export helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.app.lib.tree_explorer import build_tree_export_png, build_tree_export_svg, build_tree_exports
from jarl.experiments.graph import ExperimentGraph
from jarl.training.config import RLRunConfig


def _build_small_tree(exp_dir: Path) -> ExperimentGraph[RLRunConfig]:
    graph = ExperimentGraph(exp_dir, base_config=RLRunConfig())
    root = graph.create_root(RLRunConfig(), branch="main", label="root", prepare=True)
    graph.extend("main", label="child", prepare=True)
    graph.fork("alt", from_node=root, label="fork", prepare=True)
    graph.save()
    return ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)


class TestBuildTreeExportSvg:
    """Tests for build_tree_export_svg."""

    def test_vertical_export_is_git_tree_svg(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph = _build_small_tree(exp_dir)
        root_id = next(node_id for node_id, degree in graph.as_networkx().in_degree() if degree == 0)

        svg = build_tree_export_svg(
            exp_dir,
            metric_key="eval/episode_return",
            style="vertical",
        ).decode("utf-8")

        assert svg.startswith("<?xml")
        assert 'data-layout="vertical"' in svg
        assert f'data-node-id="{root_id}"' in svg
        assert 'data-edge="fork"' in svg
        assert ">ROOT<" in svg

    def test_horizontal_export_is_git_tree_svg(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph = _build_small_tree(exp_dir)
        root_id = next(node_id for node_id, degree in graph.as_networkx().in_degree() if degree == 0)

        svg = build_tree_export_svg(
            exp_dir,
            metric_key="eval/episode_return",
            style="horizontal",
        ).decode("utf-8")

        assert 'data-layout="horizontal"' in svg
        assert f'data-node-id="{root_id}"' in svg
        assert 'data-edge="fork"' in svg

    def test_dag_export_is_matplotlib_svg(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        _build_small_tree(exp_dir)

        svg = build_tree_export_svg(
            exp_dir,
            metric_key="eval/episode_return",
            style="dag",
        ).decode("utf-8")

        assert "<svg" in svg
        assert "matplotlib" in svg

    def test_unknown_style_raises(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        _build_small_tree(exp_dir)

        with pytest.raises(ValueError, match="Unknown tree export style"):
            build_tree_export_svg(
                exp_dir,
                metric_key="eval/episode_return",
                style="radial",  # type: ignore[arg-type]
            )


class TestBuildTreeExportPng:
    """Tests for PNG export helpers."""

    @pytest.mark.parametrize(
        "style",
        [
            pytest.param("vertical", id="vertical"),
            pytest.param("horizontal", id="horizontal"),
            pytest.param("dag", id="dag"),
        ],
    )
    def test_png_signature(self, tmp_path: Path, style: str) -> None:
        exp_dir = tmp_path / "exp"
        _build_small_tree(exp_dir)

        png = build_tree_export_png(
            exp_dir,
            metric_key="eval/episode_return",
            style=style,  # type: ignore[arg-type]
        )

        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(png) > 100

    def test_exports_pair_matches_individual_builders(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        _build_small_tree(exp_dir)

        svg, png = build_tree_exports(exp_dir, metric_key="eval/episode_return", style="horizontal")

        assert svg == build_tree_export_svg(exp_dir, metric_key="eval/episode_return", style="horizontal")
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(png) > 100
