"""Tests for git-style tree SVG and PNG export."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.config import BaseConfig
from jarl.experiments.git_tree_svg import BRANCH_COLORS, build_git_tree_png, build_git_tree_svg
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.viz import build_git_tree_png as reexported_png
from jarl.experiments.viz import build_git_tree_svg as reexported_svg


class SampleConfig(BaseConfig):
    """Concrete config for git-tree export tests."""

    value: float = 1.0


@pytest.fixture()
def graph(tmp_path: Path) -> ExperimentGraph[SampleConfig]:
    """Tree with an extend on main and a fork from root."""
    exp_dir = tmp_path / "exp"
    built: ExperimentGraph[SampleConfig] = ExperimentGraph(exp_dir)
    root = built.create_root(config=SampleConfig(), branch="main", label="root")
    child = built.extend("main", label="step2")
    built.fork("alt", from_node=root, label="branch")
    root.save_metrics({"test_accuracy": 0.5})
    child.save_metrics({"test_accuracy": 0.9})
    return built


class TestBuildGitTreeSvg:
    """Tests for build_git_tree_svg."""

    def test_empty_graph_raises(self, tmp_path: Path) -> None:
        graph: ExperimentGraph[SampleConfig] = ExperimentGraph(tmp_path / "empty")

        with pytest.raises(ValueError, match="no nodes"):
            build_git_tree_svg(graph)

    def test_unknown_orientation_raises(self, graph: ExperimentGraph[SampleConfig]) -> None:
        with pytest.raises(ValueError, match="Unknown git tree orientation"):
            build_git_tree_svg(graph, orientation="radial")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "orientation", [pytest.param("vertical", id="vertical"), pytest.param("horizontal", id="horizontal")]
    )
    def test_svg_marks_git_structure(self, graph: ExperimentGraph[SampleConfig], orientation: str) -> None:
        root_id = next(node_id for node_id, degree in graph.as_networkx().in_degree() if degree == 0)
        child_id = next(
            node_id
            for node_id, workspace in graph.all_nodes.items()
            if workspace.branch == "main" and node_id != root_id
        )
        fork_id = next(node_id for node_id, workspace in graph.all_nodes.items() if workspace.branch == "alt")

        svg = build_git_tree_svg(graph, metric_key="test_accuracy", orientation=orientation)  # type: ignore[arg-type]

        assert svg.startswith("<?xml")
        assert f'data-layout="{orientation}"' in svg
        assert f'data-node-id="{root_id}"' in svg
        assert f'data-node-id="{child_id}"' in svg
        assert f'data-node-id="{fork_id}"' in svg
        assert 'data-edge="extend"' in svg
        assert 'data-edge="fork"' in svg
        assert ">ROOT<" in svg
        assert ">HEAD<" in svg
        assert 'data-current="true"' in svg
        assert BRANCH_COLORS[0] in svg
        assert 'stroke-dasharray="6 4"' in _path_tag(svg, edge="fork")
        assert "stroke-dasharray" not in _path_tag(svg, edge="extend")

    def test_horizontal_places_children_to_the_right(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root_id = next(node_id for node_id, degree in graph.as_networkx().in_degree() if degree == 0)
        child_id = next(
            node_id
            for node_id, workspace in graph.all_nodes.items()
            if workspace.branch == "main" and node_id != root_id
        )

        svg = build_git_tree_svg(graph, metric_key="test_accuracy", orientation="horizontal")

        assert float(_group_attr(svg, root_id, "data-x")) < float(_group_attr(svg, child_id, "data-x"))

    def test_vertical_places_children_below(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root_id = next(node_id for node_id, degree in graph.as_networkx().in_degree() if degree == 0)
        child_id = next(
            node_id
            for node_id, workspace in graph.all_nodes.items()
            if workspace.branch == "main" and node_id != root_id
        )

        svg = build_git_tree_svg(graph, metric_key="test_accuracy", orientation="vertical")

        assert float(_group_attr(svg, root_id, "data-y")) < float(_group_attr(svg, child_id, "data-y"))

    def test_reexported_from_viz(self) -> None:
        assert reexported_svg is build_git_tree_svg
        assert reexported_png is build_git_tree_png


class TestBuildGitTreePng:
    """Tests for build_git_tree_png."""

    @pytest.mark.parametrize(
        "orientation", [pytest.param("vertical", id="vertical"), pytest.param("horizontal", id="horizontal")]
    )
    def test_png_signature(self, graph: ExperimentGraph[SampleConfig], orientation: str) -> None:
        png = build_git_tree_png(graph, metric_key="test_accuracy", orientation=orientation)  # type: ignore[arg-type]

        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(png) > 100


def _path_tag(svg: str, *, edge: str) -> str:
    """Return the `<path>` tag marked with the given edge kind."""
    marker = f'data-edge="{edge}"'
    start = svg.rfind("<path", 0, svg.index(marker))
    end = svg.find("/>", svg.index(marker)) + 2
    return svg[start:end]


def _group_attr(svg: str, node_id: str, attr: str) -> str:
    """Return one attribute from the SVG group for ``node_id``."""
    marker = f'data-node-id="{node_id}"'
    start = svg.rfind("<g ", 0, svg.index(marker) + len(marker))
    end = svg.find(">", svg.index(marker)) + 1
    tag = svg[start:end]
    prefix = f'{attr}="'
    value_start = tag.index(prefix) + len(prefix)
    value_end = tag.index('"', value_start)
    return tag[value_start:value_end]
