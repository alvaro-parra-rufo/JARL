"""Experiment tree explorer module."""

from __future__ import annotations

from jarl.app.lib.layout import require_experiment_dir
from jarl.app.lib.session import init_session_state
from jarl.app.lib.sidebar import render_sidebar
from jarl.app.lib.tree_page import render_tree_page
from jarl.app.lib.ui import configure_page, page_header

configure_page("Árbol")
init_session_state({})
render_sidebar()

page_header(
    "Árbol",
    "Explora el DAG del experimento con zoom, búsqueda y actualización live.",
)

exp_dir = require_experiment_dir(empty_title="Sin experimento activo")
if exp_dir is not None:
    render_tree_page(exp_dir)
