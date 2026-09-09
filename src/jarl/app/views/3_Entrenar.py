"""Training module: new experiment, launch, continue, and debug."""

from __future__ import annotations

from jarl.app.lib.form_state import apply_pending_form_mutations, init_persisted_form
from jarl.app.lib.layout import render_module_tab_selector
from jarl.app.lib.session import experiment_dir, init_session_state
from jarl.app.lib.sidebar import render_sidebar
from jarl.app.lib.training_page import (
    render_training_continue,
    render_training_debug,
    render_training_launch,
    render_training_new,
)
from jarl.app.lib.tree_explorer import TRAINING_TAB_SESSION_KEY
from jarl.app.lib.ui import configure_page, empty_state, page_header

TRAINING_TABS = ("Nuevo", "Lanzar", "Continuar", "Debug")

configure_page("Entrenamiento")
init_session_state({TRAINING_TAB_SESSION_KEY: "Lanzar"})
init_persisted_form()

exp_dir = experiment_dir()
apply_pending_form_mutations(exp_dir)

render_sidebar()

page_header(
    "Entrenamiento",
    "Crea experimentos, lanza o reanuda nodos y continúa el linaje.",
)

active_tab = render_module_tab_selector(
    TRAINING_TABS,
    session_key=TRAINING_TAB_SESSION_KEY,
    default="Lanzar",
)

if active_tab == "Nuevo":
    render_training_new()
elif active_tab == "Lanzar":
    if exp_dir is None:
        empty_state("Sin experimento activo", "Selecciona uno en la barra lateral o crea uno en **Nuevo**.")
    else:
        render_training_launch(exp_dir)
elif active_tab == "Continuar":
    if exp_dir is None:
        empty_state("Sin experimento activo", "Selecciona uno en la barra lateral o crea uno en **Nuevo**.")
    else:
        render_training_continue(exp_dir)
elif active_tab == "Debug":
    if exp_dir is None:
        empty_state("Sin experimento activo", "Selecciona uno en la barra lateral.")
    else:
        render_training_debug(exp_dir)
