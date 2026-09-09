"""Run reusable agentic evaluation cases from Runner Lab."""

from __future__ import annotations

import streamlit as st

from jarl.app.lib.session import init_session_state
from jarl.app.lib.sidebar import render_sidebar
from jarl.app.lib.ui import configure_page, empty_state, page_header
from jarl.utils.extras import is_extra_available

configure_page("Testing")
init_session_state({"active_agentic_case_run": None})
render_sidebar()

page_header(
    "Testing agéntico",
    "Ejecuta casos reproducibles o revisa conversaciones, tools y artefactos de runs anteriores.",
)

if not is_extra_available("agentic"):
    empty_state(
        "Extra agentic no instalado",
        "Instala `jarl[agentic]` para listar y ejecutar casos agénticos.",
    )
    st.stop()

from jarl.app.lib.testing_page import render_testing_page  # noqa: E402

render_testing_page()
