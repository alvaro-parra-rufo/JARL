"""Page render helpers for the Runner Lab metrics module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

from jarl.app.lib.inspection_panel import render_inspection_panel
from jarl.app.lib.layout import select_node_workspace
from jarl.app.lib.live_ui import render_live_metrics_fragment
from jarl.app.lib.metrics_view import (
    DEFAULT_METRIC_PRESET,
    METRIC_PRESETS,
    build_multi_node_frame,
    build_multi_node_latest_table,
    comparison_metric_options,
    default_metric_choices,
    default_metric_prefixes,
    downsample_frame,
    metrics_file_cache_key,
    preset_metric_names,
    snapshot_to_wide_frame,
    tail_frame,
)
from jarl.app.lib.session import active_run, list_node_summaries
from jarl.app.lib.ui import empty_state, format_float, path_block
from jarl.experiments.metrics import (
    MetricsSnapshot,
    build_metrics_snapshot,
    concatenate_metric_snapshots,
    describe_lineage_concat,
    extend_chain_nodes,
)

if TYPE_CHECKING:
    from jarl.experiments.graph import ExperimentGraph
    from jarl.experiments.node import NodeWorkspace
    from jarl.training.config import RLRunConfig

DEFAULT_MAX_CHART_ROWS = 2_000
DEFAULT_MAX_TABLE_ROWS = 1_000
MIN_LINEAGE_CHAIN_NODES = 2

METRICAS_PRESET_KEY = "metricas_preset"
METRICAS_SELECTED_KEY = "metricas_selected"
METRICAS_APPLIED_PRESET_KEY = "metricas_applied_preset"
METRICAS_PREFIXES_KEY = "metricas_prefix_groups"
METRICAS_NODE_CONTEXT_KEY = "metricas_active_node"
METRICAS_COMPARE_NODES_KEY = "metricas_compare_nodes"
METRICAS_COMPARE_METRICS_KEY = "metricas_compare_metrics"
METRICAS_LINEAGE_NODES_KEY = "metricas_lineage_nodes"
METRICAS_LINEAGE_CHAIN_IDS_KEY = "metricas_lineage_chain_ids"
METRICAS_LINEAGE_METRICS_KEY = "metricas_lineage_metrics"
METRICAS_LINEAGE_PRESET_KEY = "metricas_lineage_preset"
METRICAS_LINEAGE_APPLIED_KEY = "metricas_lineage_applied_preset"

__all__ = [
    "METRICAS_APPLIED_PRESET_KEY",
    "METRICAS_COMPARE_METRICS_KEY",
    "METRICAS_COMPARE_NODES_KEY",
    "METRICAS_LINEAGE_APPLIED_KEY",
    "METRICAS_LINEAGE_CHAIN_IDS_KEY",
    "METRICAS_LINEAGE_METRICS_KEY",
    "METRICAS_LINEAGE_NODES_KEY",
    "METRICAS_NODE_CONTEXT_KEY",
    "METRICAS_PREFIXES_KEY",
    "METRICAS_PRESET_KEY",
    "METRICAS_SELECTED_KEY",
    "cached_metrics_snapshot",
    "render_metrics_compare",
    "render_metrics_explorer",
    "render_metrics_inspection",
    "render_metrics_lineage_concat",
]


@st.cache_data(show_spinner="Cargando métricas…")
def cached_metrics_snapshot(cache_key: tuple[str, int, int]) -> MetricsSnapshot:
    """Load metrics once per file revision (path, mtime, size)."""
    return build_metrics_snapshot(Path(cache_key[0]))


def load_node_snapshots(
    graph: ExperimentGraph[RLRunConfig],
    node_ids: list[str],
) -> dict[str, MetricsSnapshot]:
    """Load cached snapshots for multiple nodes."""
    snapshots: dict[str, MetricsSnapshot] = {}
    for node_id in node_ids:
        path = graph.get_node(node_id).metrics_jsonl_path
        snapshots[node_id] = cached_metrics_snapshot(metrics_file_cache_key(path))
    return snapshots


def render_metrics_explorer(
    workspace: NodeWorkspace,
    snapshot: MetricsSnapshot,
    *,
    metrics_path: Path,
) -> None:
    """Render single-node metrics explorer with optional live KPI fragment."""
    path_block(metrics_path)

    if snapshot.record_count == 0:
        empty_state("Sin métricas", "El nodo aún no tiene metrics.jsonl.")
        return

    if active_run() is not None:
        st.subheader("KPIs en vivo")
        render_live_metrics_fragment(
            metrics_path,
            fragment_key=f"metricas_live_{workspace.id}",
        )
    else:
        _render_static_kpis(snapshot)

    names = list(snapshot.names)
    name_set = set(names)
    prefixes = sorted({name.split("/", maxsplit=1)[0] for name in names})
    preset_options = [*METRIC_PRESETS, "Personalizado"]

    if METRICAS_PREFIXES_KEY not in st.session_state:
        st.session_state[METRICAS_PREFIXES_KEY] = default_metric_prefixes(prefixes)
    if METRICAS_PRESET_KEY not in st.session_state:
        st.session_state[METRICAS_PRESET_KEY] = DEFAULT_METRIC_PRESET

    controls = st.columns((1.1, 1.2, 1, 1))
    with controls[0]:
        preset = st.selectbox("Preset", options=preset_options, key=METRICAS_PRESET_KEY)
    with controls[1]:
        selected_prefixes = st.multiselect("Grupos", options=prefixes, key=METRICAS_PREFIXES_KEY)
    with controls[2]:
        max_chart_rows = st.number_input(
            "Puntos gráfico",
            min_value=100,
            max_value=50_000,
            value=DEFAULT_MAX_CHART_ROWS,
            step=100,
        )
    with controls[3]:
        max_table_rows = st.number_input(
            "Filas raw",
            min_value=100,
            max_value=50_000,
            value=DEFAULT_MAX_TABLE_ROWS,
            step=100,
        )

    filtered_names = [name for name in names if name.split("/", maxsplit=1)[0] in selected_prefixes]
    if preset != "Personalizado":
        preset_default = [name for name in preset_metric_names(preset, name_set) if name in filtered_names]
    else:
        preset_default = [name for name in default_metric_choices(snapshot.latest) if name in filtered_names]

    if st.session_state.get(METRICAS_APPLIED_PRESET_KEY) != preset:
        st.session_state[METRICAS_APPLIED_PRESET_KEY] = preset
        st.session_state[METRICAS_SELECTED_KEY] = preset_default

    selected = st.multiselect(
        "Métricas",
        options=filtered_names,
        key=METRICAS_SELECTED_KEY,
    )
    selected = [name for name in dict.fromkeys(selected) if name in name_set]
    _render_metrics_result_tabs(
        snapshot,
        selected=selected,
        max_chart_rows=int(max_chart_rows),
        max_table_rows=int(max_table_rows),
    )


def _render_metrics_result_tabs(
    snapshot: MetricsSnapshot,
    *,
    selected: list[str],
    max_chart_rows: int,
    max_table_rows: int,
) -> None:
    """Render chart, latest-values, and raw tabs for the metrics explorer."""
    chart_tab, latest_tab, raw_tab = st.tabs(("Gráfico", "Últimos valores", "Raw"))
    latest = snapshot.latest
    with chart_tab:
        if selected:
            chart_frame = downsample_frame(
                snapshot_to_wide_frame(snapshot, selected),
                max_rows=max_chart_rows,
            )
            if len(chart_frame) < snapshot.step_count:
                st.caption(
                    f"Mostrando {len(chart_frame):,} de {snapshot.step_count:,} steps para mantener la UI fluida."
                )
            st.line_chart(chart_frame)
        else:
            empty_state("Sin selección", "Elige métricas para dibujar.")
    with latest_tab:
        latest_rows = [{"metric": name, "value": value} for name, value in sorted(latest.items())]
        st.dataframe(latest_rows, use_container_width=True, hide_index=True)
    with raw_tab:
        if not selected:
            empty_state("Sin selección", "Elige métricas para ver filas raw.")
            return
        full_raw = snapshot_to_wide_frame(snapshot, selected)
        raw_frame = tail_frame(full_raw, max_rows=max_table_rows)
        if len(raw_frame) < len(full_raw):
            st.caption(f"Mostrando últimas {len(raw_frame):,} filas de {len(full_raw):,}.")
        st.dataframe(raw_frame, use_container_width=True)


def render_metrics_compare(
    graph: ExperimentGraph[RLRunConfig],
    exp_dir: Path,
    *,
    default_node_id: str,
) -> None:
    """Render multi-node metric comparison charts and latest-value table."""
    summaries = list_node_summaries(exp_dir, graph=graph)
    node_ids = [node_id for node_id, _ in summaries]
    if not node_ids:
        empty_state("Sin nodos", "Crea nodos con métricas para comparar.")
        return

    default_nodes = [default_node_id]
    for head in graph.branch_heads.values():
        if head.id in node_ids and head.id not in default_nodes:
            default_nodes.append(head.id)

    if METRICAS_COMPARE_NODES_KEY not in st.session_state:
        st.session_state[METRICAS_COMPARE_NODES_KEY] = default_nodes[: min(4, len(default_nodes))]

    controls = st.columns((1.4, 1, 1))
    with controls[0]:
        selected_nodes = st.multiselect(
            "Nodos",
            options=node_ids,
            key=METRICAS_COMPARE_NODES_KEY,
        )
    with controls[1]:
        max_chart_rows = st.number_input(
            "Puntos gráfico",
            min_value=100,
            max_value=50_000,
            value=DEFAULT_MAX_CHART_ROWS,
            step=100,
            key="metricas_compare_chart_rows",
        )
    with controls[2]:
        compare_preset = st.selectbox(
            "Preset",
            options=[*METRIC_PRESETS, "Personalizado"],
            index=0,
            key="metricas_compare_preset",
        )

    if not selected_nodes:
        empty_state("Sin nodos", "Selecciona al menos un nodo.")
        return

    snapshots = load_node_snapshots(graph, selected_nodes)
    metric_options = comparison_metric_options(snapshots)
    if not metric_options:
        empty_state("Sin métricas", "Los nodos seleccionados no tienen metrics.jsonl.")
        return

    if compare_preset != "Personalizado":
        preset_default = preset_metric_names(compare_preset, set(metric_options))
    else:
        preset_default = [name for name in metric_options if name.startswith(("rollout/", "eval/"))][:2]
        preset_default = preset_default or metric_options[:2]

    if st.session_state.get("metricas_compare_applied_preset") != compare_preset:
        st.session_state["metricas_compare_applied_preset"] = compare_preset
        st.session_state[METRICAS_COMPARE_METRICS_KEY] = preset_default

    if METRICAS_COMPARE_METRICS_KEY not in st.session_state:
        st.session_state[METRICAS_COMPARE_METRICS_KEY] = preset_default

    selected_metrics = st.multiselect(
        "Métricas",
        options=metric_options,
        key=METRICAS_COMPARE_METRICS_KEY,
    )
    if not selected_metrics:
        empty_state("Sin métricas", "Elige al menos una métrica para comparar.")
        return

    chart_frame = downsample_frame(
        build_multi_node_frame(snapshots, selected_metrics),
        max_rows=int(max_chart_rows),
    )
    total_steps = max((snapshot.step_count for snapshot in snapshots.values()), default=0)
    if len(chart_frame) < total_steps:
        st.caption(f"Mostrando {len(chart_frame):,} puntos (downsample) para mantener la UI fluida.")
    st.line_chart(chart_frame)

    latest_table = build_multi_node_latest_table(snapshots, selected_metrics)
    st.subheader("Últimos valores")
    st.dataframe(latest_table, use_container_width=True, hide_index=True)


def render_metrics_lineage_concat(
    graph: ExperimentGraph[RLRunConfig],
    exp_dir: Path,
    *,
    default_node_id: str,
) -> None:
    """Concatenate metrics along an extend chain (same branch) into one timeline."""
    workspace = select_node_workspace(
        graph,
        exp_dir,
        session_key="metricas_lineage_end_node",
        label="Nodo final (extends)",
        default_node_id=default_node_id,
    )
    chain = extend_chain_nodes(graph, workspace.id)
    if len(chain) < MIN_LINEAGE_CHAIN_NODES:
        empty_state(
            "Sin cadena extend",
            "Selecciona un nodo con al menos un extend en la misma rama. "
            "Los forks a otra rama no se concatenan automáticamente.",
        )
        return

    chain_ids = [node.id for node in chain]
    if st.session_state.get(METRICAS_LINEAGE_NODES_KEY) != workspace.id:
        st.session_state[METRICAS_LINEAGE_NODES_KEY] = workspace.id
        st.session_state[METRICAS_LINEAGE_APPLIED_KEY] = DEFAULT_METRIC_PRESET
        st.session_state[METRICAS_LINEAGE_PRESET_KEY] = DEFAULT_METRIC_PRESET
        st.session_state.pop(METRICAS_LINEAGE_CHAIN_IDS_KEY, None)
        st.session_state.pop(METRICAS_LINEAGE_METRICS_KEY, None)

    if METRICAS_LINEAGE_CHAIN_IDS_KEY not in st.session_state:
        st.session_state[METRICAS_LINEAGE_CHAIN_IDS_KEY] = chain_ids

    selected_ids = st.multiselect(
        "Nodos en la cadena",
        options=chain_ids,
        format_func=lambda node_id: node_id,
        key=METRICAS_LINEAGE_CHAIN_IDS_KEY,
        help="Orden root → final. Cada tramo suma el ``parent_checkpoint_step`` del hijo.",
    )
    ordered_ids = [node_id for node_id in chain_ids if node_id in selected_ids]
    if len(ordered_ids) < MIN_LINEAGE_CHAIN_NODES:
        empty_state("Selecciona al menos dos nodos consecutivos en la cadena.")
        return

    segments = [(graph.get_node(node_id), load_node_snapshots(graph, [node_id])[node_id]) for node_id in ordered_ids]
    if all(snapshot.record_count == 0 for _, snapshot in segments):
        empty_state("Sin métricas", "Los nodos seleccionados no tienen metrics.jsonl.")
        return

    merged = concatenate_metric_snapshots(segments)
    st.caption(" · ".join(describe_lineage_concat(segments)))

    names = list(merged.names)
    name_set = set(names)
    preset_options = [*METRIC_PRESETS, "Personalizado"]
    if METRICAS_LINEAGE_PRESET_KEY not in st.session_state:
        st.session_state[METRICAS_LINEAGE_PRESET_KEY] = DEFAULT_METRIC_PRESET

    controls = st.columns((1.1, 1, 1))
    with controls[0]:
        preset = st.selectbox("Preset", options=preset_options, key=METRICAS_LINEAGE_PRESET_KEY)
    with controls[1]:
        max_chart_rows = st.number_input(
            "Puntos gráfico",
            min_value=100,
            max_value=50_000,
            value=DEFAULT_MAX_CHART_ROWS,
            step=100,
            key="metricas_lineage_chart_rows",
        )
    with controls[2]:
        max_table_rows = st.number_input(
            "Filas raw",
            min_value=100,
            max_value=50_000,
            value=DEFAULT_MAX_TABLE_ROWS,
            step=100,
            key="metricas_lineage_table_rows",
        )

    if preset != "Personalizado":
        preset_default = preset_metric_names(preset, name_set)
    else:
        preset_default = default_metric_choices(merged.latest)

    lineage_metrics_key = METRICAS_LINEAGE_METRICS_KEY
    if st.session_state.get(METRICAS_LINEAGE_APPLIED_KEY) != preset:
        st.session_state[METRICAS_LINEAGE_APPLIED_KEY] = preset
        st.session_state[lineage_metrics_key] = preset_default

    if lineage_metrics_key not in st.session_state:
        st.session_state[lineage_metrics_key] = preset_default

    selected = st.multiselect(
        "Métricas",
        options=names,
        key=lineage_metrics_key,
    )
    selected = [name for name in dict.fromkeys(selected) if name in name_set]

    _render_static_kpis(merged)
    _render_metrics_result_tabs(
        merged,
        selected=selected,
        max_chart_rows=int(max_chart_rows),
        max_table_rows=int(max_table_rows),
    )


def render_metrics_inspection(
    graph: ExperimentGraph[RLRunConfig],
    exp_dir: Path,
) -> None:
    """Render inspection tools; train/showcase vídeos only (infer_* → Inferencias)."""
    render_inspection_panel(
        graph,
        exp_dir,
        include_videos=True,
        exclude_inference_videos=True,
        inspect_session_key="metricas_inspection_node",
    )


def _render_static_kpis(snapshot: MetricsSnapshot) -> None:
    latest = snapshot.latest
    top_metrics = [
        ("Train return", latest.get("rollout/episode_return")),
        ("Eval return", latest.get("eval/episode_return")),
        ("SPS", latest.get("time/sps")),
        ("Env steps", latest.get("steps/nr_env_steps")),
    ]
    cols = st.columns(6)
    for col, (label, value) in zip(cols, top_metrics, strict=False):
        col.metric(label, format_float(value))
    cols[4].metric("Records", f"{snapshot.record_count:,}")
    cols[5].metric("Steps", f"{snapshot.step_count:,}")
