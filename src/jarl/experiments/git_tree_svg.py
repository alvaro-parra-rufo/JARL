"""SVG and PNG export of the experiment tree in git-log style."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from math import pi
from typing import Literal
from xml.sax.saxutils import escape

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeWorkspace

__all__ = [
    "BRANCH_COLORS",
    "GitTreeOrientation",
    "build_git_tree_exports",
    "build_git_tree_png",
    "build_git_tree_svg",
]

GitTreeOrientation = Literal["vertical", "horizontal"]

BRANCH_COLORS: tuple[str, ...] = (
    "#74d88f",
    "#58c7f3",
    "#f2a65a",
    "#d66ba0",
    "#b98cff",
    "#5eead4",
    "#ff8a7a",
    "#c7d95b",
    "#7cc7ff",
    "#f4c95d",
    "#9bd48b",
    "#f48fb1",
    "#80cbc4",
    "#ffb74d",
    "#90caf9",
    "#ce93d8",
    "#a5d6a7",
    "#ffab91",
    "#b0bec5",
    "#e6ee9c",
    "#81d4fa",
    "#bcaaa4",
    "#ffd54f",
    "#b39ddb",
)
"""Branch palette matching the live tree explorer."""

_EdgeKind = Literal["root", "extend", "fork"]

_BG = "#0f1117"
_PANEL = "#1a1d24"
_BORDER = "#2d3340"
_TEXT = "#f5f7fb"
_MUTED = "#9aa3b2"
_CURRENT = "#f0b429"
_HEAD = "#d66ba0"

_MARGIN_X = 28.0
_MARGIN_Y = 24.0
_LEGEND_H = 44.0
_V_INDENT = 56.0
_H_GAP = 56.0
_SIBLING_GAP = 14.0
_CARD_W_VERTICAL = 560.0
_CARD_W_HORIZONTAL = 360.0
_CARD_H = 96.0
_V_GAP = 18.0
_DOT_R = 6.0
_ROW_DOT_X = 8.0
_CARD_X_OFFSET = 28.0
_PNG_DPI = 144.0
_HSL_LIGHTNESS_MID = 0.5
"""Lightness threshold in the HSL to RGB conversion."""


@dataclass(frozen=True, slots=True)
class _PlacedNode:
    """Laid-out node used while emitting SVG and PNG primitives."""

    node_id: str
    x: float
    y: float
    depth: int
    card_w: float
    parent_id: str | None
    edge_kind: _EdgeKind
    branch: str
    branch_color: str
    parent_branch_color: str
    label: str
    status: str
    step: int
    metric_value: float | None
    is_root: bool
    is_head: bool
    is_current: bool
    is_fork_source: bool

    @property
    def dot_cx(self) -> float:
        """X center of the git-style node dot."""
        return self.x + _ROW_DOT_X

    @property
    def dot_cy(self) -> float:
        """Y center of the git-style node dot."""
        return self.y + 28.0

    @property
    def card_x(self) -> float:
        """Left edge of the node card."""
        return self.x + _CARD_X_OFFSET

    @property
    def card_y(self) -> float:
        """Top edge of the node card."""
        return self.y

    @property
    def card_right(self) -> float:
        """Right edge of the node card."""
        return self.card_x + self.card_w


@dataclass(frozen=True, slots=True)
class _TreeLayout:
    """Geometry and labels for one exported git-style tree."""

    placed: dict[str, _PlacedNode]
    children: dict[str, tuple[str, ...]]
    width: float
    height: float
    metric_key: str
    metric_min: float | None
    metric_max: float | None
    orientation: GitTreeOrientation


def build_git_tree_svg(
    graph: ExperimentGraph,
    *,
    metric_key: str = "rollout/episode_return",
    orientation: GitTreeOrientation = "vertical",
) -> str:
    """Render the full experiment tree as a git-style SVG.

    The layout matches the live explorer: vertical (root on top, children
    indented) or horizontal (children to the right, siblings stacked). Fork vs
    extend edges and the branch palette match the app.

    Args:
        graph: Experiment graph to visualize.
        metric_key: Metric shown on each card and used for the card accent.
        orientation: ``vertical`` or ``horizontal`` explorer layout.

    Returns:
        UTF-8 SVG document.

    Raises:
        ValueError: If the graph has no nodes, more than one root, or an unknown orientation.
    """
    return _svg_from_layout(_build_layout(graph, metric_key=metric_key, orientation=orientation))


def build_git_tree_png(
    graph: ExperimentGraph,
    *,
    metric_key: str = "rollout/episode_return",
    orientation: GitTreeOrientation = "vertical",
) -> bytes:
    """Render the full experiment tree as a git-style PNG.

    Args:
        graph: Experiment graph to visualize.
        metric_key: Metric shown on each card and used for the card accent.
        orientation: ``vertical`` or ``horizontal`` explorer layout.

    Returns:
        PNG document bytes.

    Raises:
        ValueError: If the graph has no nodes, more than one root, or an unknown orientation.
    """
    return _png_from_layout(_build_layout(graph, metric_key=metric_key, orientation=orientation))


def build_git_tree_exports(
    graph: ExperimentGraph,
    *,
    metric_key: str = "rollout/episode_return",
    orientation: GitTreeOrientation = "vertical",
) -> tuple[str, bytes]:
    """Render SVG and PNG from a single layout pass.

    Args:
        graph: Experiment graph to visualize.
        metric_key: Metric shown on each card and used for the card accent.
        orientation: ``vertical`` or ``horizontal`` explorer layout.

    Returns:
        Pair of UTF-8 SVG document and PNG bytes.
    """
    layout = _build_layout(graph, metric_key=metric_key, orientation=orientation)
    return _svg_from_layout(layout), _png_from_layout(layout)


def _build_layout(
    graph: ExperimentGraph,
    *,
    metric_key: str,
    orientation: GitTreeOrientation,
) -> _TreeLayout:
    """Compute node positions for the requested explorer orientation."""
    if orientation not in ("vertical", "horizontal"):
        msg = f"Unknown git tree orientation: {orientation!r}"
        raise ValueError(msg)

    root_id = _require_single_root(graph)
    children = _children_by_parent(graph)
    visit_order = _preorder(root_id, children)
    branch_colors = _branch_colors(graph.get_node(node_id).branch for node_id in visit_order)
    metric_values = {node_id: _node_metric(graph.get_node(node_id), metric_key) for node_id in visit_order}
    present_values = [value for value in metric_values.values() if value is not None]
    placed = _place_nodes(
        graph,
        root_id=root_id,
        children=children,
        branch_colors=branch_colors,
        metric_values=metric_values,
        orientation=orientation,
    )
    return _TreeLayout(
        placed=placed,
        children=children,
        width=max(node.card_right for node in placed.values()) + _MARGIN_X,
        height=max(node.y + _CARD_H for node in placed.values()) + _MARGIN_Y,
        metric_key=metric_key,
        metric_min=min(present_values) if present_values else None,
        metric_max=max(present_values) if present_values else None,
        orientation=orientation,
    )


def _require_single_root(graph: ExperimentGraph) -> str:
    """Return the unique root id of a tree-shaped experiment graph."""
    tree = graph.as_networkx()
    if tree.number_of_nodes() == 0:
        msg = "Experiment graph has no nodes to plot."
        raise ValueError(msg)
    roots = [node_id for node_id, in_degree in tree.in_degree() if in_degree == 0]
    if len(roots) != 1:
        msg = f"Experiment graph must have a single root, got {roots}."
        raise ValueError(msg)
    return roots[0]


def _place_nodes(
    graph: ExperimentGraph,
    *,
    root_id: str,
    children: dict[str, tuple[str, ...]],
    branch_colors: dict[str, str],
    metric_values: dict[str, float | None],
    orientation: GitTreeOrientation,
) -> dict[str, _PlacedNode]:
    """Assign canvas coordinates to every node in the tree."""
    head_ids = {workspace.id for workspace in graph.branch_heads.values()}
    current_id = graph.current_node.id
    card_w = _CARD_W_VERTICAL if orientation == "vertical" else _CARD_W_HORIZONTAL
    placed: dict[str, _PlacedNode] = {}
    origin_y = _MARGIN_Y + _LEGEND_H

    def add_node(node_id: str, *, x: float, y: float, depth: int) -> _PlacedNode:
        """Record one placed node and return it."""
        workspace = graph.get_node(node_id)
        parent_id = workspace.node_metadata.parent_id
        parent_ws = graph.get_node(parent_id) if parent_id else None
        branch_color = branch_colors[workspace.branch]
        parent_branch_color = branch_colors[parent_ws.branch] if parent_ws is not None else branch_color
        child_ids = children.get(node_id, ())
        node = _PlacedNode(
            node_id=node_id,
            x=x,
            y=y,
            depth=depth,
            card_w=card_w,
            parent_id=parent_id,
            edge_kind=_edge_kind(workspace, parent_ws),
            branch=workspace.branch,
            branch_color=branch_color,
            parent_branch_color=parent_branch_color,
            label=workspace.node_metadata.label or "—",
            status=workspace.status.value,
            step=workspace.node_metadata.step,
            metric_value=metric_values[node_id],
            is_root=node_id == root_id,
            is_head=node_id in head_ids,
            is_current=node_id == current_id,
            is_fork_source=any(graph.get_node(child_id).branch != workspace.branch for child_id in child_ids),
        )
        placed[node_id] = node
        return node

    if orientation == "vertical":

        def place_vertical(node_id: str, depth: int, y: float) -> float:
            """Place ``node_id`` and descendants in a top-down git layout."""
            add_node(node_id, x=_MARGIN_X + depth * _V_INDENT, y=y, depth=depth)
            y_next = y + _CARD_H + _V_GAP
            for child_id in children.get(node_id, ()):
                y_next = place_vertical(child_id, depth + 1, y_next)
            return y_next

        place_vertical(root_id, 0, origin_y)
        return placed

    def place_horizontal(node_id: str, x: float, y: float, depth: int) -> float:
        """Place ``node_id`` and descendants in a left-to-right git layout."""
        add_node(node_id, x=x, y=y, depth=depth)
        child_ids = children.get(node_id, ())
        if not child_ids:
            return _CARD_H
        child_x = x + _CARD_X_OFFSET + card_w + _H_GAP
        child_y = y
        heights: list[float] = []
        for child_id in child_ids:
            height = place_horizontal(child_id, child_x, child_y, depth + 1)
            heights.append(height)
            child_y += height + _SIBLING_GAP
        return max(_CARD_H, sum(heights) + _SIBLING_GAP * (len(heights) - 1))

    place_horizontal(root_id, _MARGIN_X, origin_y, 0)
    return placed


def _svg_from_layout(layout: _TreeLayout) -> str:
    """Serialize a laid-out git tree to an SVG document."""
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{layout.width:.0f}" height="{layout.height:.0f}" '
            f'viewBox="0 0 {layout.width:.1f} {layout.height:.1f}" data-layout="{layout.orientation}" role="img">'
        ),
        f"<title>Experiment tree ({_esc(layout.metric_key)})</title>",
        f'<rect width="100%" height="100%" fill="{_BG}"/>',
        _legend_svg(),
        *_edge_svgs(layout),
        *(
            _node_svg(node, metric_key=layout.metric_key, metric_min=layout.metric_min, metric_max=layout.metric_max)
            for node in layout.placed.values()
        ),
        "</svg>",
    ]
    return "\n".join(parts)


def _png_from_layout(layout: _TreeLayout) -> bytes:
    """Rasterize a laid-out git tree with matplotlib."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    fig_w = max(layout.width / _PNG_DPI, 1.0)
    fig_h = max(layout.height / _PNG_DPI, 1.0)
    figure = Figure(figsize=(fig_w, fig_h), dpi=_PNG_DPI, facecolor=_BG)
    FigureCanvasAgg(figure)
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.set_xlim(0.0, layout.width)
    axes.set_ylim(layout.height, 0.0)
    axes.set_axis_off()
    axes.set_facecolor(_BG)
    axes.add_patch(Rectangle((0.0, 0.0), layout.width, layout.height, facecolor=_BG, edgecolor="none", zorder=0))
    _draw_legend_png(axes)
    _draw_edges_png(axes, layout)
    for node in layout.placed.values():
        _draw_node_png(axes, node, layout)
        _draw_dot_png(axes, node)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=_PNG_DPI, facecolor=_BG)
    buffer.seek(0)
    png = buffer.getvalue()
    figure.clear()
    return png


def _esc(text: str) -> str:
    """Escape text for XML element and attribute content."""
    return escape(text, {"'": "&apos;", '"': "&quot;"})


def _node_metric(workspace: NodeWorkspace, metric_key: str) -> float | None:
    """Return one latest scalar metric for a node card."""
    value = workspace.latest_metrics().get(metric_key)
    return float(value) if value is not None else None


def _children_by_parent(graph: ExperimentGraph) -> dict[str, tuple[str, ...]]:
    """Return sorted child ids grouped by parent id."""
    children: dict[str, list[str]] = {}
    for parent_id, child_id in graph.as_networkx().edges():
        children.setdefault(parent_id, []).append(child_id)
    return {parent_id: tuple(sorted(child_ids)) for parent_id, child_ids in children.items()}


def _preorder(root_id: str, children: dict[str, tuple[str, ...]]) -> list[str]:
    """Walk the tree root-first so branch colors match first-seen order in the explorer."""
    order: list[str] = []

    def visit(node_id: str) -> None:
        """Append ``node_id`` then visit children in sorted order."""
        order.append(node_id)
        for child_id in children.get(node_id, ()):
            visit(child_id)

    visit(root_id)
    return order


def _branch_colors(branches: Iterable[str]) -> dict[str, str]:
    """Assign explorer palette colors in first-seen branch order, pinning ``main`` to green."""
    ordered: list[str] = []
    seen: set[str] = set()
    for branch in branches:
        if branch not in seen:
            seen.add(branch)
            ordered.append(branch)

    colors: dict[str, str] = {}
    next_index = 1 if "main" in seen else 0
    for branch in ordered:
        if branch == "main":
            colors[branch] = BRANCH_COLORS[0]
            continue
        colors[branch] = BRANCH_COLORS[next_index % len(BRANCH_COLORS)]
        next_index += 1
    return colors


def _edge_kind(workspace: NodeWorkspace, parent: NodeWorkspace | None) -> _EdgeKind:
    """Classify the incoming edge the same way the live explorer does."""
    if parent is None:
        return "root"
    return "extend" if parent.branch == workspace.branch else "fork"


def _format_metric(name: str, value: float | None) -> str:
    """Format a metric badge like the live explorer."""
    if value is None:
        return f"{name}: —"
    return f"{name}: {value:.4f}"


def _hsl_to_rgb(hue: float, saturation: float, lightness: float) -> tuple[float, float, float]:
    """Convert HSL in unit interval to RGB in unit interval."""
    if saturation == 0.0:
        return (lightness, lightness, lightness)
    q = (
        lightness * (1.0 + saturation)
        if lightness < _HSL_LIGHTNESS_MID
        else lightness + saturation - lightness * saturation
    )
    p = 2.0 * lightness - q
    return (
        _hue_to_channel(p, q, hue + 1.0 / 3.0),
        _hue_to_channel(p, q, hue),
        _hue_to_channel(p, q, hue - 1.0 / 3.0),
    )


def _hue_to_channel(p: float, q: float, t: float) -> float:
    """HSL helper that maps one hue offset onto a single RGB channel."""
    tone = t % 1.0
    if tone < 1.0 / 6.0:
        return p + (q - p) * 6.0 * tone
    if tone < 1.0 / 2.0:
        return q
    if tone < 2.0 / 3.0:
        return p + (q - p) * (2.0 / 3.0 - tone) * 6.0
    return p


def _metric_rgba(
    value: float | None, metric_min: float | None, metric_max: float | None
) -> tuple[float, float, float, float]:
    """Return the explorer metric accent as an RGBA tuple."""
    if value is None or metric_min is None or metric_max is None or metric_min == metric_max:
        return (91 / 255, 141 / 255, 239 / 255, 0.25)
    t = (value - metric_min) / (metric_max - metric_min)
    hue = (210.0 - t * 150.0) / 360.0
    red, green, blue = _hsl_to_rgb(hue, 0.70, 0.45)
    return (red, green, blue, 0.28)


def _metric_fill(value: float | None, metric_min: float | None, metric_max: float | None) -> str:
    """Return the explorer metric accent color for a card footer bar."""
    red, green, blue, alpha = _metric_rgba(value, metric_min, metric_max)
    return f"rgba({round(red * 255)},{round(green * 255)},{round(blue * 255)},{alpha})"


def _hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` plus alpha to an SVG ``rgba()`` color."""
    red, green, blue, _ = _hex_rgba_tuple(hex_color, alpha)
    return f"rgba({round(red * 255)},{round(green * 255)},{round(blue * 255)},{alpha})"


def _hex_rgba_tuple(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Convert ``#rrggbb`` plus alpha to an RGBA tuple."""
    normalized = hex_color.removeprefix("#")
    red = int(normalized[0:2], 16) / 255.0
    green = int(normalized[2:4], 16) / 255.0
    blue = int(normalized[4:6], 16) / 255.0
    return (red, green, blue, alpha)


def _pt(pixels: float) -> float:
    """Convert SVG pixel sizes to matplotlib points at the PNG dpi."""
    return pixels * 72.0 / _PNG_DPI


def _legend_svg() -> str:
    """Return the EXTEND / FORK legend matching the live explorer."""
    y = _MARGIN_Y
    return (
        f'<g data-legend="true" font-family="Segoe UI, system-ui, sans-serif" font-size="11">'
        f'<rect x="{_MARGIN_X}" y="{y}" width="220" height="28" rx="8" fill="rgba(26,29,36,0.78)" stroke="{_BORDER}"/>'
        f'<line x1="{_MARGIN_X + 14}" y1="{y + 14}" x2="{_MARGIN_X + 48}" y2="{y + 14}" '
        f'stroke="{_MUTED}" stroke-width="2"/>'
        f'<text x="{_MARGIN_X + 54}" y="{y + 18}" fill="{_TEXT}">EXTEND</text>'
        f'<line x1="{_MARGIN_X + 118}" y1="{y + 14}" x2="{_MARGIN_X + 152}" y2="{y + 14}" '
        f'stroke="{_MUTED}" stroke-width="2" stroke-dasharray="6 4"/>'
        f'<text x="{_MARGIN_X + 158}" y="{y + 18}" fill="{_TEXT}">FORK</text>'
        f"</g>"
    )


def _edge_svgs(layout: _TreeLayout) -> list[str]:
    """Emit git-style connectors for the current orientation."""
    if layout.orientation == "horizontal":
        return _horizontal_edge_svgs(layout)
    return _vertical_edge_svgs(layout)


def _vertical_edge_svgs(layout: _TreeLayout) -> list[str]:
    """Emit vertical spines and horizontal elbows for every parent/child pair."""
    parts: list[str] = []
    for parent_id, child_ids in layout.children.items():
        if not child_ids:
            continue
        parent = layout.placed[parent_id]
        last = layout.placed[child_ids[-1]]
        spine_x = parent.dot_cx
        parts.append(
            f'<line data-spine="{_esc(parent_id)}" x1="{spine_x:.1f}" y1="{parent.dot_cy:.1f}" '
            f'x2="{spine_x:.1f}" y2="{last.dot_cy:.1f}" stroke="{parent.branch_color}" '
            f'stroke-width="3" stroke-linecap="round" stroke-dasharray="1 9" opacity="0.72"/>'
        )
        for child_id in child_ids:
            child = layout.placed[child_id]
            dashed = ' stroke-dasharray="6 4"' if child.edge_kind == "fork" else ""
            parts.append(
                f'<path data-edge="{child.edge_kind}" data-from="{_esc(parent_id)}" data-to="{_esc(child_id)}" '
                f'd="M {spine_x:.1f} {child.dot_cy:.1f} L {child.dot_cx:.1f} {child.dot_cy:.1f}" '
                f'stroke="{child.parent_branch_color}" stroke-width="2.5" fill="none"{dashed}/>'
            )
    return parts


def _horizontal_edge_svgs(layout: _TreeLayout) -> list[str]:
    """Emit rightward spines and elbows matching the explorer horizontal layout."""
    parts: list[str] = []
    for parent_id, child_ids in layout.children.items():
        if not child_ids:
            continue
        parent = layout.placed[parent_id]
        first = layout.placed[child_ids[0]]
        last = layout.placed[child_ids[-1]]
        spine_x = first.x - 22.0
        parts.append(
            f'<line data-spine-stub="{_esc(parent_id)}" x1="{parent.card_right:.1f}" y1="{parent.dot_cy:.1f}" '
            f'x2="{spine_x:.1f}" y2="{parent.dot_cy:.1f}" stroke="{parent.branch_color}" '
            f'stroke-width="2.5" stroke-linecap="round" opacity="0.86"/>'
        )
        if first.dot_cy != last.dot_cy:
            parts.append(
                f'<line data-spine="{_esc(parent_id)}" x1="{spine_x:.1f}" y1="{first.dot_cy:.1f}" '
                f'x2="{spine_x:.1f}" y2="{last.dot_cy:.1f}" stroke="{parent.branch_color}" '
                f'stroke-width="3" stroke-linecap="round" stroke-dasharray="1 9" opacity="0.85"/>'
            )
        for child_id in child_ids:
            child = layout.placed[child_id]
            dashed = ' stroke-dasharray="6 4"' if child.edge_kind == "fork" else ""
            parts.append(
                f'<path data-edge="{child.edge_kind}" data-from="{_esc(parent_id)}" data-to="{_esc(child_id)}" '
                f'd="M {spine_x:.1f} {child.dot_cy:.1f} L {child.dot_cx:.1f} {child.dot_cy:.1f}" '
                f'stroke="{child.parent_branch_color}" stroke-width="2.5" fill="none"{dashed}/>'
            )
    return parts


def _node_badges(node: _PlacedNode, *, metric_key: str) -> list[_Badge]:
    """Return the status and lineage badges drawn on a node card."""
    badges = [_badge(node.status, fill=_status_fill(node.status))]
    if node.is_root:
        badges.append(_badge("ROOT", fill="rgba(116, 216, 143, 0.22)", stroke=node.branch_color))
    if node.edge_kind != "root":
        badges.append(_badge(node.edge_kind, fill="rgba(255,255,255,0.06)"))
    if node.is_fork_source:
        badges.append(_badge("FORKS", fill="rgba(242, 166, 90, 0.18)"))
    if node.is_head:
        badges.append(_badge("HEAD", fill="rgba(214, 107, 160, 0.22)", stroke=_HEAD))
    badges.append(_badge(_format_metric(metric_key, node.metric_value)))
    return badges


def _badge_positions(node: _PlacedNode, badges: list[_Badge]) -> list[tuple[_Badge, float, float]]:
    """Pack badges left-to-right, wrapping when they overflow the card."""
    badge_x = node.card_x + 14
    badge_y = node.card_y + 68
    placed: list[tuple[_Badge, float, float]] = []
    for badge in badges:
        if badge_x + badge.width > node.card_x + node.card_w - 10 and placed:
            badge_x = node.card_x + 14
            badge_y += 16
        placed.append((badge, badge_x, badge_y))
        badge_x += badge.width + 6
    return placed


def _node_svg(
    node: _PlacedNode,
    *,
    metric_key: str,
    metric_min: float | None,
    metric_max: float | None,
) -> str:
    """Emit one git-style node: dot, card, badges."""
    metric_color = _metric_fill(node.metric_value, metric_min, metric_max)
    border = _CURRENT if node.is_current else _BORDER
    soft = _hex_rgba(node.branch_color, 0.18)
    badge_parts = [badge.at(x, y) for badge, x, y in _badge_positions(node, _node_badges(node, metric_key=metric_key))]

    if node.edge_kind == "fork":
        dot = (
            f'<rect data-dot="fork" x="{node.dot_cx - 5:.1f}" y="{node.dot_cy - 5:.1f}" width="10" height="10" '
            f'rx="2" fill="{node.branch_color}" stroke="{_BG}" stroke-width="2" '
            f'transform="rotate(45 {node.dot_cx:.1f} {node.dot_cy:.1f})"/>'
        )
    else:
        dot = (
            f'<circle data-dot="{node.edge_kind}" cx="{node.dot_cx:.1f}" cy="{node.dot_cy:.1f}" r="{_DOT_R}" '
            f'fill="{node.branch_color}" stroke="{_BG}" stroke-width="2"/>'
        )

    current_attr = ' data-current="true"' if node.is_current else ""
    meta = f"{node.branch} · {node.label} · {node.status} · step {node.step}"
    return (
        f'<g data-node-id="{_esc(node.node_id)}" data-branch="{_esc(node.branch)}" '
        f'data-edge-kind="{node.edge_kind}" data-x="{node.x:.1f}" data-y="{node.y:.1f}"{current_attr} '
        f'font-family="Segoe UI, system-ui, sans-serif">'
        f"{dot}"
        f'<rect x="{node.card_x:.1f}" y="{node.card_y:.1f}" width="{node.card_w:.1f}" height="{_CARD_H:.1f}" '
        f'rx="10" fill="{_PANEL}" stroke="{border}" stroke-width="{2 if node.is_current else 1}"/>'
        f'<rect x="{node.card_x:.1f}" y="{node.card_y:.1f}" width="4" height="{_CARD_H:.1f}" '
        f'fill="{node.branch_color}"/>'
        f'<rect x="{node.card_x:.1f}" y="{node.card_y + _CARD_H - 4:.1f}" width="{node.card_w:.1f}" height="4" '
        f'fill="{metric_color}"/>'
        f'<rect x="{node.card_x:.1f}" y="{node.card_y:.1f}" width="{node.card_w * 0.46:.1f}" height="{_CARD_H:.1f}" '
        f'fill="{soft}" opacity="0.9"/>'
        f'<text x="{node.card_x + 14:.1f}" y="{node.card_y + 22:.1f}" fill="{_TEXT}" font-size="13" '
        f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">{_esc(node.node_id)}</text>'
        f'<rect x="{node.card_x + node.card_w - 92:.1f}" y="{node.card_y + 8:.1f}" width="80" height="18" rx="9" '
        f'fill="{_hex_rgba(node.branch_color, 0.16)}" stroke="{_hex_rgba(node.branch_color, 0.55)}"/>'
        f'<text x="{node.card_x + node.card_w - 52:.1f}" y="{node.card_y + 21:.1f}" fill="{_TEXT}" font-size="10" '
        f'text-anchor="middle">{_esc(node.branch)}</text>'
        f'<text x="{node.card_x + 14:.1f}" y="{node.card_y + 42:.1f}" fill="{_MUTED}" font-size="11">'
        f"{_esc(meta)}</text>"
        f"{''.join(badge_parts)}"
        f"</g>"
    )


@dataclass(frozen=True, slots=True)
class _Badge:
    """Measured SVG badge used to pack labels on a card."""

    label: str
    fill: str
    stroke: str
    width: float

    def at(self, x: float, y: float) -> str:
        """Render the badge with its top-left at ``(x, y)``."""
        return (
            f"<g>"
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{self.width:.1f}" height="16" rx="8" '
            f'fill="{self.fill}" stroke="{self.stroke}"/>'
            f'<text x="{x + self.width / 2:.1f}" y="{y + 12:.1f}" fill="{_TEXT}" font-size="9" '
            f'text-anchor="middle">{_esc(self.label)}</text>'
            f"</g>"
        )


def _badge(label: str, fill: str = "rgba(255,255,255,0.06)", stroke: str = _BORDER) -> _Badge:
    """Build a compact badge sized to its label."""
    width = max(36.0, 8.0 + 6.2 * len(label))
    return _Badge(label=label, fill=fill, stroke=stroke, width=width)


def _status_fill(status: str) -> str:
    """Return a soft fill for a node status badge."""
    fills = {
        "completed": "rgba(62, 207, 142, 0.22)",
        "training": "rgba(91, 141, 239, 0.22)",
        "failed": "rgba(239, 107, 107, 0.22)",
        "interrupted": "rgba(242, 166, 90, 0.18)",
        "prepared": "rgba(158, 167, 178, 0.18)",
        "created": "rgba(158, 167, 178, 0.12)",
    }
    return fills.get(status, "rgba(255,255,255,0.06)")


def _parse_css_color(color: str) -> str | tuple[float, float, float, float]:
    """Return a matplotlib-compatible color from an SVG fill/stroke string."""
    if color.startswith("rgba(") and color.endswith(")"):
        body = color.removeprefix("rgba(").removesuffix(")")
        red_s, green_s, blue_s, alpha_s = [part.strip() for part in body.split(",")]
        return (int(red_s) / 255.0, int(green_s) / 255.0, int(blue_s) / 255.0, float(alpha_s))
    return color


def _draw_legend_png(axes: object) -> None:
    """Draw the EXTEND / FORK legend on a matplotlib axes."""
    from matplotlib.axes import Axes
    from matplotlib.patches import FancyBboxPatch

    if not isinstance(axes, Axes):
        msg = "PNG legend requires matplotlib axes."
        raise TypeError(msg)
    y = _MARGIN_Y
    axes.add_patch(
        FancyBboxPatch(
            (_MARGIN_X, y),
            220.0,
            28.0,
            boxstyle="round,pad=0,rounding_size=8",
            facecolor=(26 / 255, 29 / 255, 36 / 255, 0.78),
            edgecolor=_BORDER,
            linewidth=_pt(1),
            zorder=2,
        )
    )
    axes.plot(
        [_MARGIN_X + 14, _MARGIN_X + 48],
        [y + 14, y + 14],
        color=_MUTED,
        linewidth=_pt(2),
        solid_capstyle="round",
        zorder=3,
    )
    axes.text(_MARGIN_X + 54, y + 18, "EXTEND", color=_TEXT, fontsize=_pt(11), va="center", zorder=3)
    axes.plot(
        [_MARGIN_X + 118, _MARGIN_X + 152],
        [y + 14, y + 14],
        color=_MUTED,
        linewidth=_pt(2),
        linestyle=(0, (6, 4)),
        solid_capstyle="round",
        zorder=3,
    )
    axes.text(_MARGIN_X + 158, y + 18, "FORK", color=_TEXT, fontsize=_pt(11), va="center", zorder=3)


def _draw_edges_png(axes: object, layout: _TreeLayout) -> None:
    """Draw git-style connectors on a matplotlib axes."""
    from matplotlib.axes import Axes

    if not isinstance(axes, Axes):
        msg = "PNG edges require matplotlib axes."
        raise TypeError(msg)
    for parent_id, child_ids in layout.children.items():
        if not child_ids:
            continue
        parent = layout.placed[parent_id]
        first = layout.placed[child_ids[0]]
        last = layout.placed[child_ids[-1]]
        if layout.orientation == "vertical":
            spine_x = parent.dot_cx
            axes.plot(
                [spine_x, spine_x],
                [parent.dot_cy, last.dot_cy],
                color=_hex_rgba_tuple(parent.branch_color, 0.72),
                linewidth=_pt(3),
                linestyle=(0, (1, 9)),
                solid_capstyle="round",
                zorder=1,
            )
        else:
            spine_x = first.x - 22.0
            axes.plot(
                [parent.card_right, spine_x],
                [parent.dot_cy, parent.dot_cy],
                color=_hex_rgba_tuple(parent.branch_color, 0.86),
                linewidth=_pt(2.5),
                solid_capstyle="round",
                zorder=1,
            )
            if first.dot_cy != last.dot_cy:
                axes.plot(
                    [spine_x, spine_x],
                    [first.dot_cy, last.dot_cy],
                    color=_hex_rgba_tuple(parent.branch_color, 0.85),
                    linewidth=_pt(3),
                    linestyle=(0, (1, 9)),
                    solid_capstyle="round",
                    zorder=1,
                )
        for child_id in child_ids:
            child = layout.placed[child_id]
            linestyle: str | tuple[int, tuple[int, int]] = (0, (6, 4)) if child.edge_kind == "fork" else "solid"
            axes.plot(
                [spine_x, child.dot_cx],
                [child.dot_cy, child.dot_cy],
                color=child.parent_branch_color,
                linewidth=_pt(2.5),
                linestyle=linestyle,
                zorder=1,
            )


def _draw_node_png(axes: object, node: _PlacedNode, layout: _TreeLayout) -> None:
    """Draw one node card and its labels on a matplotlib axes."""
    from matplotlib.axes import Axes
    from matplotlib.patches import FancyBboxPatch, Rectangle

    if not isinstance(axes, Axes):
        msg = "PNG nodes require matplotlib axes."
        raise TypeError(msg)
    border = _CURRENT if node.is_current else _BORDER
    axes.add_patch(
        FancyBboxPatch(
            (node.card_x, node.card_y),
            node.card_w,
            _CARD_H,
            boxstyle="round,pad=0,rounding_size=10",
            facecolor=_PANEL,
            edgecolor=border,
            linewidth=_pt(2 if node.is_current else 1),
            zorder=2,
        )
    )
    axes.add_patch(
        Rectangle((node.card_x, node.card_y), 4.0, _CARD_H, facecolor=node.branch_color, edgecolor="none", zorder=3)
    )
    axes.add_patch(
        Rectangle(
            (node.card_x, node.card_y + _CARD_H - 4.0),
            node.card_w,
            4.0,
            facecolor=_metric_rgba(node.metric_value, layout.metric_min, layout.metric_max),
            edgecolor="none",
            zorder=3,
        )
    )
    axes.add_patch(
        Rectangle(
            (node.card_x, node.card_y),
            node.card_w * 0.46,
            _CARD_H,
            facecolor=_hex_rgba_tuple(node.branch_color, 0.18),
            edgecolor="none",
            zorder=3,
        )
    )
    axes.text(
        node.card_x + 14.0,
        node.card_y + 14.0,
        node.node_id,
        color=_TEXT,
        fontsize=_pt(13),
        fontfamily="monospace",
        va="center",
        zorder=4,
    )
    axes.add_patch(
        FancyBboxPatch(
            (node.card_x + node.card_w - 92.0, node.card_y + 8.0),
            80.0,
            18.0,
            boxstyle="round,pad=0,rounding_size=9",
            facecolor=_hex_rgba_tuple(node.branch_color, 0.16),
            edgecolor=_hex_rgba_tuple(node.branch_color, 0.55),
            linewidth=_pt(1),
            zorder=4,
        )
    )
    axes.text(
        node.card_x + node.card_w - 52.0,
        node.card_y + 17.0,
        node.branch,
        color=_TEXT,
        fontsize=_pt(10),
        ha="center",
        va="center",
        zorder=5,
    )
    meta = f"{node.branch} · {node.label} · {node.status} · step {node.step}"
    axes.text(node.card_x + 14.0, node.card_y + 38.0, meta, color=_MUTED, fontsize=_pt(11), va="center", zorder=4)
    for badge, x, y in _badge_positions(node, _node_badges(node, metric_key=layout.metric_key)):
        axes.add_patch(
            FancyBboxPatch(
                (x, y),
                badge.width,
                16.0,
                boxstyle="round,pad=0,rounding_size=8",
                facecolor=_parse_css_color(badge.fill),
                edgecolor=_parse_css_color(badge.stroke),
                linewidth=_pt(1),
                zorder=4,
            )
        )
        axes.text(
            x + badge.width / 2.0,
            y + 8.0,
            badge.label,
            color=_TEXT,
            fontsize=_pt(9),
            ha="center",
            va="center",
            zorder=5,
        )


def _draw_dot_png(axes: object, node: _PlacedNode) -> None:
    """Draw the git-style commit marker for one node."""
    from matplotlib.axes import Axes
    from matplotlib.patches import Circle, RegularPolygon

    if not isinstance(axes, Axes):
        msg = "PNG dots require matplotlib axes."
        raise TypeError(msg)
    if node.edge_kind == "fork":
        axes.add_patch(
            RegularPolygon(
                (node.dot_cx, node.dot_cy),
                4,
                radius=_DOT_R * 1.15,
                orientation=pi / 4,
                facecolor=node.branch_color,
                edgecolor=_BG,
                linewidth=_pt(2),
                zorder=6,
            )
        )
        return
    axes.add_patch(
        Circle(
            (node.dot_cx, node.dot_cy),
            _DOT_R,
            facecolor=node.branch_color,
            edgecolor=_BG,
            linewidth=_pt(2),
            zorder=6,
        )
    )
