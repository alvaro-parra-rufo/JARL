"""Shared Streamlit sidebar for the Runner Lab."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from jarl.app.lib.debug.flag import MODEL_DEBUG_FLAG_KEY
from jarl.app.lib.form_state import init_persisted_form
from jarl.app.lib.live_ui import render_sidebar_run_fragment
from jarl.app.lib.project_env import load_project_env
from jarl.app.lib.session import (
    apply_pending_node_picker_sync,
    checkout_experiment_node,
    experiment_dir,
    init_session_state,
    list_experiments,
    list_node_summaries,
    set_experiment_dir,
    set_workspace_root,
    workspace_root,
)
from jarl.app.lib.workspace_presets import (
    WORKSPACE_PRESET_CUSTOM,
    infer_workspace_preset,
    resolve_custom_workspace_target,
    resolve_workspace_preset_root,
    sticky_workspace_preset,
    workspace_preset_labels,
    workspace_preset_options,
)

__all__ = ["render_sidebar"]

_WORKSPACE_PRESET_KEY = "jarl_workspace_preset"
_WORKSPACE_CUSTOM_PATH_KEY = "jarl_workspace_custom_path"


def render_sidebar() -> None:
    """Render workspace controls and active-run status."""
    load_project_env()
    init_session_state(
        {
            "experiment_dir": None,
            "active_run": None,
            "active_inference_run": None,
            "active_agentic_case_run": None,
        }
    )
    apply_pending_node_picker_sync()
    init_persisted_form()
    with st.sidebar:
        st.header("JARL Runner Lab")
        st.caption("Experimentos Navix locales")

        st.subheader("Workspace")
        _render_workspace_presets()

        experiments = list_experiments()
        experiments_by_name = {path.name: path for path in experiments}
        current = experiment_dir()
        current_name = current.name if current is not None else "— ninguno —"
        options = ["— ninguno —", *experiments_by_name]
        if current is not None and current_name not in options and (current / "experiment.json").is_file():
            options.append(current_name)
            experiments_by_name[current_name] = current
        selected = st.selectbox(
            "Experimento activo",
            options=options,
            index=options.index(current_name) if current_name in options else 0,
        )
        if selected != "— ninguno —":
            set_experiment_dir(experiments_by_name[selected])
        else:
            set_experiment_dir(None)

        current = experiment_dir()
        if current is not None:
            st.caption("Activo")
            st.code(str(current), language=None)
            _render_checkout(current)

        st.divider()
        st.subheader("Run")
        render_sidebar_run_fragment(
            testing_page_path="testing",
            training_page_path="entrenar",
        )
        st.toggle(
            "Debug del modelo",
            key=MODEL_DEBUG_FLAG_KEY,
            help="Prompts y tools que ve el LLM, anclados a cada tramo del grafo.",
        )


def _render_workspace_presets() -> None:
    """Render preset workspace selection with optional custom path."""
    labels = workspace_preset_labels()
    options = workspace_preset_options()
    st.session_state[_WORKSPACE_PRESET_KEY] = sticky_workspace_preset(
        inferred=infer_workspace_preset(workspace_root()),
        selected=st.session_state.get(_WORKSPACE_PRESET_KEY),
    )
    if _WORKSPACE_CUSTOM_PATH_KEY not in st.session_state:
        st.session_state[_WORKSPACE_CUSTOM_PATH_KEY] = str(workspace_root())

    st.selectbox(
        "Directorio base",
        options=options,
        format_func=lambda preset: labels[preset],
        key=_WORKSPACE_PRESET_KEY,
        on_change=_on_workspace_preset_change,
    )

    if st.session_state[_WORKSPACE_PRESET_KEY] == WORKSPACE_PRESET_CUSTOM:
        with st.form("jarl_workspace_custom_form", border=False):
            st.text_input(
                "Ruta personalizada",
                key=_WORKSPACE_CUSTOM_PATH_KEY,
                help=(
                    "Carpeta padre de experimentos (p. ej. results/cmp) o la carpeta "
                    "del experimento (la que contiene experiment.json)."
                ),
            )
            submitted = st.form_submit_button("Aplicar ruta", icon=":material/folder_managed:")
        if submitted and _apply_workspace_preset():
            st.rerun()
    else:
        st.caption(labels[st.session_state[_WORKSPACE_PRESET_KEY]])
    st.caption(f"Aplicado: `{workspace_root()}`")


def _on_workspace_preset_change() -> None:
    """Apply Dags/Cases immediately; leave Personalizado until the path is submitted."""
    preset = str(st.session_state[_WORKSPACE_PRESET_KEY])
    if preset == WORKSPACE_PRESET_CUSTOM:
        return
    set_workspace_root(resolve_workspace_preset_root(preset))


def _apply_workspace_preset() -> bool:
    """Persist the workspace directory for the selected preset.

    Returns:
        True when the workspace was updated.
    """
    preset = str(st.session_state[_WORKSPACE_PRESET_KEY])
    if preset == WORKSPACE_PRESET_CUSTOM:
        custom_path = str(st.session_state.get(_WORKSPACE_CUSTOM_PATH_KEY, ""))
        try:
            target = resolve_custom_workspace_target(custom_path)
        except ValueError as exc:
            st.error(str(exc))
            return False
        set_workspace_root(target.root)
        if target.experiment_dir is not None:
            set_experiment_dir(target.experiment_dir)
        return True
    set_workspace_root(resolve_workspace_preset_root(preset))
    return True


def _render_checkout(exp_dir: Path) -> None:
    from jarl.app.lib.session import load_graph

    graph = load_graph(exp_dir)
    node_ids = [node_id for node_id, _ in list_node_summaries(exp_dir)]
    if not node_ids:
        return
    if "sidebar_checkout_node" not in st.session_state:
        st.session_state["sidebar_checkout_node"] = graph.current_node.id
    selected = st.selectbox(
        "Checkout nodo",
        options=node_ids,
        format_func=lambda node_id: node_id,
        key="sidebar_checkout_node",
    )
    if selected != graph.current_node.id and st.button("Aplicar checkout", icon=":material/target:"):
        checkout_experiment_node(exp_dir, selected)
        st.rerun()
