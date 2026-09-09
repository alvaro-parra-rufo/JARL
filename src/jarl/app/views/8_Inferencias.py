"""Launch greedy inference (checkpoint + environment + seed) for Runner Lab."""

from __future__ import annotations

import streamlit as st

from jarl.app.lib.inference_page import LAST_INFERENCE_JOB_KEY, render_inference_page
from jarl.app.lib.layout import load_experiment_graph, require_experiment_dir
from jarl.app.lib.session import init_session_state
from jarl.app.lib.sidebar import render_sidebar
from jarl.app.lib.ui import configure_page, page_header

configure_page("Inferencias")
init_session_state({LAST_INFERENCE_JOB_KEY: None})
render_sidebar()

page_header(
    "Lanzar inferencia",
    "Evalúa un checkpoint en un entorno con la seed que elijas. "
    "El vídeo y los returns del rollout aparecen aquí al terminar — no en Métricas.",
)

exp_dir = require_experiment_dir()
if exp_dir is None:
    st.stop()

graph = load_experiment_graph(exp_dir)
render_inference_page(graph, exp_dir)
