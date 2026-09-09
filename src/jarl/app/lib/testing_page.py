"""Runner Lab presentation for agentic case evaluation."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from datetime import datetime
from pathlib import Path

import streamlit as st

from jarl.agentic.cases import (
    AgenticCaseInfo,
    AgenticCaseLaunchRequest,
    AgenticCaseResult,
    agentic_case_log_path,
    filter_agentic_case_runs,
    list_agentic_case_runs,
    list_agentic_cases,
    load_agentic_case_result,
    load_favorite_run_names,
    set_favorite_run,
)
from jarl.agentic.cases.labels import agentic_case_exit_code_label, agentic_case_status_label
from jarl.agentic.errors import SessionError
from jarl.agentic.llm import (
    MAIN_COMPONENT_ID,
    list_packaged_llm_catalog_paths,
    load_llm_catalog,
    resolve_llm_yaml_path,
)
from jarl.agentic.progress import progress_path
from jarl.app.lib.agentic_case_bridge import AGENTIC_PROGRESS_BUFFER_KEY, launch_agentic_case
from jarl.app.lib.conversation import (
    poll_conversation_transcript,
    reset_conversation_live_cache,
    track_conversation_experiment_dir,
)
from jarl.app.lib.debug.panel import render_conversation
from jarl.app.lib.layout import render_module_tab_selector
from jarl.app.lib.live_feed import (
    cursor_key_for_path,
    read_log_delta,
    update_terminal_buffer,
)
from jarl.app.lib.live_ui import (
    AGENTIC_CASE_FINISH_NOTIFIED_KEY,
    AGENTIC_CONVERSATION_WAKE_KEY,
    AGENTIC_LAUNCH_UNBLOCK_KEY,
    AGENTIC_LIVE_WATCH_KEY,
    AGENTIC_RESULT_WATCH_KEY,
    DEFAULT_LOG_REFRESH_SECONDS,
    DEFAULT_RUN_STATUS_REFRESH_SECONDS,
    LAST_AGENTIC_CASE_EXIT_CODE_KEY,
)
from jarl.app.lib.navigation import nav_page_path
from jarl.app.lib.session import (
    active_agentic_case_run,
    set_experiment_dir,
    set_workspace_root,
)
from jarl.experiments.cases import allocate_case_destination
from jarl.experiments.paths import resolve_cases_root

LAST_AGENTIC_CASE_DIR_KEY = "jarl_last_agentic_case_dir"
"""Session key for the last launched agentic case experiment directory."""

TESTING_TAB_SESSION_KEY = "testing_tab"
"""Streamlit session key for the Testing module tab selector."""

TESTING_TABS = ("Ejecutar", "Revisar")
"""Visible Testing module tabs."""

_REVIEW_STATUSES = ("passed", "failed", "error")
"""Persisted statuses offered in the historical run filter."""

__all__ = [
    "LAST_AGENTIC_CASE_DIR_KEY",
    "TESTING_TABS",
    "TESTING_TAB_SESSION_KEY",
    "render_testing_page",
]


def render_testing_page() -> None:
    """Render the execute catalog or the historical run replay."""
    _ensure_cases_workspace()
    active_tab = render_module_tab_selector(
        TESTING_TABS,
        session_key=TESTING_TAB_SESSION_KEY,
        default="Ejecutar",
    )
    if active_tab == "Revisar":
        _render_review_section()
        return
    _render_launch_section()


def _render_launch_section() -> None:
    """Render the case catalog, launcher, live log, and validation result."""
    cases = list_agentic_cases()
    if not cases:
        st.info("No hay casos agénticos registrados.")
        return

    filtered = _filter_agentic_cases(cases)
    if not filtered:
        st.warning("Ningún caso coincide con los filtros.")
        return

    selected = st.selectbox(
        "Caso",
        options=filtered,
        format_func=lambda case: f"{case.title} · {case.id}",
        key="testing_agentic_case",
    )
    _render_case_detail(selected)
    settings_ready, llm_config = _render_provider_status()
    launch_options = _render_launch_options(selected)
    st.caption(f"Destino de casos: `{_cases_root_display()}`")
    _render_testing_live_fragment(selected, settings_ready, llm_config, launch_options)

    st.subheader("Resultado")
    _render_case_result_fragment()


def _filter_agentic_cases(cases: tuple[AgenticCaseInfo, ...]) -> list[AgenticCaseInfo]:
    """Filter the case catalog by tags and free-text search."""
    available_tags = sorted({tag for case in cases for tag in case.tags})
    filter_cols = st.columns((2, 3))
    with filter_cols[0]:
        selected_tags = st.multiselect(
            "Categorías",
            options=available_tags,
            default=[],
            key="testing_case_tags",
            help="Tags del caso (p. ej. env, graph, fork, read). Vacío = todas.",
        )
    with filter_cols[1]:
        query = st.text_input(
            "Buscar",
            key="testing_case_search",
            placeholder="id o título…",
        )

    filtered = list(cases)
    if selected_tags:
        required = set(selected_tags)
        filtered = [case for case in filtered if required.issubset(case.tags)]
    needle = query.strip().casefold()
    if needle:
        filtered = [
            case
            for case in filtered
            if needle in case.id.casefold()
            or needle in case.title.casefold()
            or any(needle in tag.casefold() for tag in case.tags)
        ]

    filter_fingerprint = (tuple(selected_tags), needle)
    previous = st.session_state.get("testing_case_filter_fp")
    if previous != filter_fingerprint:
        st.session_state["testing_case_filter_fp"] = filter_fingerprint
        # Drop stale selectbox value when the visible catalog changes.
        st.session_state.pop("testing_agentic_case", None)

    if filtered:
        st.caption(f"{len(filtered)} / {len(cases)} casos")
    return filtered


def _render_review_section() -> None:
    """Render the historical case-run picker and a finished-run replay."""
    try:
        runs = list_agentic_case_runs()
        favorite_names = load_favorite_run_names()
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        st.error(f"No se pudo listar las ejecuciones: {exc}")
        return
    if not runs:
        st.info("No hay ejecuciones con `agentic_case_result.json` en el destino de casos.")
        st.caption(f"Destino de casos: `{_cases_root_display()}`")
        return

    tags_by_case_id = _catalog_tags_by_case_id()
    filtered = _filter_review_runs(
        runs,
        tags_by_case_id=tags_by_case_id,
        favorite_names=favorite_names,
    )
    if not filtered:
        st.warning("Ninguna ejecución coincide con los filtros.")
        return

    by_name = {run.experiment_dir.name: run for run in filtered}
    selected_name = st.selectbox(
        "Ejecución",
        options=list(by_name),
        format_func=lambda name: _format_review_run_label(
            by_name[name],
            favorite=name in favorite_names,
        ),
        key="testing_review_run",
    )
    selected = by_name[selected_name]
    _render_review_run_header(selected, favorite=selected_name in favorite_names)
    _render_finished_case(
        selected.experiment_dir,
        open_key="testing_review_open_experiment",
    )


def _catalog_tags_by_case_id() -> dict[str, tuple[str, ...]]:
    """Return catalog tags keyed by agentic case id."""
    return {case.id: case.tags for case in list_agentic_cases()}


def _filter_review_runs(
    runs: tuple[AgenticCaseResult, ...],
    *,
    tags_by_case_id: Mapping[str, Collection[str]],
    favorite_names: Collection[str],
) -> list[AgenticCaseResult]:
    """Filter persisted runs by catalog tags, status, search, and favorites."""
    available_tags = sorted({tag for run in runs for tag in tags_by_case_id.get(run.case_id, ())})
    filter_cols = st.columns((2, 2, 3))
    with filter_cols[0]:
        selected_tags = st.multiselect(
            "Categorías",
            options=available_tags,
            default=[],
            key="testing_review_tags",
            help="Tags del caso (p. ej. env, graph, fork, read). Vacío = todas.",
        )
    with filter_cols[1]:
        statuses = st.multiselect(
            "Estado",
            options=_REVIEW_STATUSES,
            format_func=agentic_case_status_label,
            default=[],
            key="testing_review_status",
            help="Vacío = todos.",
        )
    with filter_cols[2]:
        query = st.text_input(
            "Buscar",
            key="testing_review_search",
            placeholder="id, tag o carpeta…",
        )
    favorites_only = st.checkbox(
        "Solo favoritos",
        key="testing_review_favorites_only",
        help="Muestra únicamente las ejecuciones marcadas con estrella.",
    )

    filtered = filter_agentic_case_runs(
        runs,
        statuses=statuses,
        query=query,
        tags=selected_tags,
        tags_by_case_id=tags_by_case_id,
        favorite_names=favorite_names,
        favorites_only=favorites_only,
    )
    selected_key = st.session_state.get("testing_review_run")
    if selected_key not in {run.experiment_dir.name for run in filtered}:
        st.session_state.pop("testing_review_run", None)

    if filtered:
        st.caption(f"{len(filtered)} / {len(runs)} ejecuciones")
    return list(filtered)


def _render_review_run_header(run: AgenticCaseResult, *, favorite: bool) -> None:
    """Render the selected run path and favorite toggle."""
    meta_cols = st.columns((4, 1))
    meta_cols[0].caption(str(run.experiment_dir))
    label = "Quitar de favoritos" if favorite else "Añadir a favoritos"
    icon = ":material/star:" if favorite else ":material/star_border:"
    if meta_cols[1].button(label, icon=icon, key="testing_review_favorite"):
        try:
            set_favorite_run(run.experiment_dir.name, favorite=not favorite)
        except (OSError, TypeError, ValueError) as exc:
            st.error(f"No se pudo actualizar favoritos: {exc}")
            return
        st.rerun()


def _format_review_run_label(run: AgenticCaseResult, *, favorite: bool = False) -> str:
    """Return a compact selectbox label for one persisted run."""
    suffix = run.experiment_dir.name.rsplit("-", 1)[-1]
    star = "★ · " if favorite else ""
    return (
        f"{star}{agentic_case_status_label(run.status)} · {run.case_id} · "
        f"{_compact_iso_timestamp(run.finished_at)} · {suffix}"
    )


def _compact_iso_timestamp(value: str) -> str:
    """Return a short UTC stamp, or the original string if it is not ISO."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    return parsed.strftime("%Y-%m-%d %H:%M")


def _render_finished_case(experiment_dir: Path, *, open_key: str) -> None:
    """Render monitor log, conversation, and result for a completed run."""
    with st.expander("Monitor"):
        _render_static_case_log(experiment_dir)
    st.subheader("Conversación")
    _render_case_conversation(experiment_dir, running=False)
    st.subheader("Resultado")
    _render_open_experiment_button(
        experiment_dir,
        key=open_key,
        require_finished=False,
    )
    _render_loaded_case_result(experiment_dir)


def _render_static_case_log(experiment_dir: Path) -> None:
    """Render the persisted case subprocess log without tailing."""
    log_path = agentic_case_log_path(experiment_dir)
    if not log_path.is_file():
        st.caption("Sin agentic_case.log.")
        return
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if text.strip():
        st.code(text, language="text")
        return
    st.caption("Log vacío.")


def _render_case_conversation(experiment_dir: Path, *, running: bool) -> None:
    """Render the conversation pane for one experiment directory."""
    track_conversation_experiment_dir(experiment_dir)
    try:
        transcript = poll_conversation_transcript(experiment_dir)
    except (SessionError, OSError, ValueError, TypeError) as exc:
        st.error(f"No se pudo leer la conversación: {exc}")
        return
    render_conversation(experiment_dir, transcript, running=running)


def _render_loaded_case_result(experiment_dir: Path) -> None:
    """Load and render a persisted validation result, if present."""
    try:
        result = load_agentic_case_result(experiment_dir)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"No se pudo leer el resultado: {exc}")
        return
    if result is None:
        st.caption("Sin ValidationReport.")
        return
    _render_case_result_body(result)


def _render_case_result_body(result: AgenticCaseResult) -> None:
    """Render one persisted case result."""
    st.metric("Estado", agentic_case_status_label(result.status))
    if result.status == "passed":
        st.success(f"`{result.case_id}` pasó todas las validaciones.")
    elif result.status == "failed":
        st.error(f"`{result.case_id}` falló la validación determinista.")
        for failure in result.failures:
            st.write(f"- {failure}")
    else:
        st.error(result.error or "La ejecución falló sin detalle.")
    st.caption(f"Inicio: {result.started_at} · fin: {result.finished_at}")


def _cases_root_display() -> str:
    """Return the resolved cases parent directory for UI hints."""
    return str(resolve_cases_root())


def _ensure_cases_workspace() -> None:
    """Align Runner Lab workspace with the cases root for tree navigation."""
    set_workspace_root(resolve_cases_root())


def _testing_live_should_tick() -> bool:
    """Return whether the unified Testing live fragment should wake on a timer."""
    return bool(
        active_agentic_case_run() is not None
        or st.session_state.get(AGENTIC_LIVE_WATCH_KEY)
        or st.session_state.get(AGENTIC_LAUNCH_UNBLOCK_KEY)
        or st.session_state.get(AGENTIC_CONVERSATION_WAKE_KEY)
    )


def _render_testing_live_fragment(
    selected: AgenticCaseInfo,
    settings_ready: bool,
    llm_config: str | None,
    launch_options: dict[str, object],
) -> None:
    """Single live fragment for launch controls, monitor tails, and conversation."""
    interval = DEFAULT_LOG_REFRESH_SECONDS if _testing_live_should_tick() else None

    @st.fragment(run_every=interval)
    def _live() -> None:
        if active_agentic_case_run() is None and st.session_state.get(AGENTIC_LAUNCH_UNBLOCK_KEY):
            st.session_state[AGENTIC_LAUNCH_UNBLOCK_KEY] = False
        active = active_agentic_case_run()
        running = active is not None
        if running:
            st.info(f"Ejecutando `{active.case_id}`…")
            st.caption(str(active.log_path))
        if st.button(
            "Ejecutar caso",
            type="primary",
            icon=":material/science:",
            disabled=running or not settings_ready,
            key="testing_launch_agentic_case",
        ):
            try:
                _ensure_cases_workspace()
                destination = allocate_case_destination(selected.id)
                request = AgenticCaseLaunchRequest(
                    case_id=selected.id,
                    destination=destination,
                    prompts=_edited_prompts(selected),
                    llm_config=llm_config,
                    launch_options=launch_options or None,
                )
                launch_agentic_case(request)
                st.session_state[LAST_AGENTIC_CASE_DIR_KEY] = str(destination)
                st.session_state[AGENTIC_CASE_FINISH_NOTIFIED_KEY] = False
                st.session_state.pop(LAST_AGENTIC_CASE_EXIT_CODE_KEY, None)
                st.session_state[AGENTIC_LIVE_WATCH_KEY] = True
                st.session_state[AGENTIC_RESULT_WATCH_KEY] = True
                st.session_state[AGENTIC_CONVERSATION_WAKE_KEY] = True
                st.session_state.pop(AGENTIC_LAUNCH_UNBLOCK_KEY, None)
                reset_conversation_live_cache()
                st.success("Caso lanzado en segundo plano.")
                st.rerun()
            except (FileExistsError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                st.error(f"No se pudo lanzar el caso: {exc}")

        st.subheader("Monitor")
        experiment_dir = _latest_experiment_dir()
        if st.session_state.get(AGENTIC_LIVE_WATCH_KEY):
            _tail_progress_feed(experiment_dir)
            _tail_case_log_feed(experiment_dir)
        _render_case_status_message()
        _render_live_activity_display(experiment_dir)
        _render_live_case_log_display(experiment_dir)

        st.subheader("Conversación")
        if experiment_dir is None:
            st.caption("Ejecuta un caso para ver la conversación.")
            return
        if not running and not st.session_state.get(AGENTIC_LIVE_WATCH_KEY):
            st.session_state[AGENTIC_CONVERSATION_WAKE_KEY] = False
        _render_case_conversation(experiment_dir, running=running)

    _live()


def _tail_progress_feed(experiment_dir: Path | None) -> None:
    """Tail progress.jsonl into the Monitor activity buffer."""
    if experiment_dir is None:
        return
    path = progress_path(experiment_dir)
    cursor_key = f"{AGENTIC_PROGRESS_BUFFER_KEY}__{cursor_key_for_path(path)}"
    raw_lines = read_log_delta(path, cursor_key=cursor_key)
    if not raw_lines:
        return
    display_lines = [_format_progress_line(line) for line in raw_lines]
    update_terminal_buffer(
        AGENTIC_PROGRESS_BUFFER_KEY,
        display_lines,
        max_lines=100,
    )


def _tail_case_log_feed(experiment_dir: Path | None) -> None:
    """Tail agentic_case.log into the Monitor log buffer."""
    if experiment_dir is None:
        return
    log_path = agentic_case_log_path(experiment_dir)
    log_cursor = f"jarl_terminal_agentic_case__{cursor_key_for_path(log_path)}"
    log_lines = read_log_delta(log_path, cursor_key=log_cursor)
    if not log_lines:
        return
    update_terminal_buffer(
        "jarl_terminal_agentic_case",
        log_lines,
        max_lines=500,
    )


def _render_live_activity_display(experiment_dir: Path | None) -> None:
    """Render the buffered progress feed."""
    st.caption("Actividad en vivo")
    if experiment_dir is None:
        st.caption("Ejecuta un caso para ver actividad en vivo.")
        return
    text = "\n".join(st.session_state.get(AGENTIC_PROGRESS_BUFFER_KEY, []))
    if text:
        st.code(text, language="text")
    else:
        st.caption("Esperando eventos de progreso…")


def _render_live_case_log_display(experiment_dir: Path | None) -> None:
    """Render the buffered case subprocess log."""
    if experiment_dir is None:
        st.caption("Esperando agentic_case.log…")
        return
    log_text = "\n".join(st.session_state.get("jarl_terminal_agentic_case", []))
    if log_text:
        st.code(log_text, language="text")
    else:
        st.caption("Esperando agentic_case.log…")


def _render_case_status_message() -> None:
    """Render subprocess status without triggering a full-page rerun."""
    active = active_agentic_case_run()
    if active is not None:
        st.info(f"Ejecutando `{active.case_id}`…")
        st.caption(str(active.log_path))
        return
    last_exit = st.session_state.get(LAST_AGENTIC_CASE_EXIT_CODE_KEY)
    if last_exit is None:
        st.caption("Sin caso activo.")
    elif int(last_exit) == 0:
        st.success("Último caso: Pasó.")
    else:
        label = agentic_case_exit_code_label(int(last_exit))
        st.warning(f"Último caso: {label}.")


def _format_progress_line(line: str) -> str:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return line
    if not isinstance(payload, dict):
        return line
    message = payload.get("message")
    if isinstance(message, str) and message:
        return message
    return line


def _render_launch_options(case: AgenticCaseInfo) -> dict[str, object]:
    """Render declared case launch options and return selected values."""
    if not case.launch_options:
        return {}
    selected: dict[str, object] = {}
    st.subheader("Opciones del caso")
    for raw_option in case.launch_options:
        if not isinstance(raw_option, dict):
            continue
        key = str(raw_option.get("key", ""))
        if not key:
            continue
        label = str(raw_option.get("label", key))
        description = str(raw_option.get("description") or "")
        option_type = str(raw_option.get("type", "bool"))
        widget_key = f"testing_launch_option__{case.id}__{key}"
        if option_type == "bool":
            default = bool(raw_option.get("default", False))
            selected[key] = st.checkbox(
                label,
                value=default,
                key=widget_key,
                help=description or None,
            )
            continue
        if option_type == "int":
            default = int(raw_option.get("default", 0))
            selected[key] = st.number_input(
                label,
                value=int(st.session_state.get(widget_key, default)),
                key=widget_key,
                help=description or None,
            )
            continue
        default = str(raw_option.get("default", ""))
        choices = raw_option.get("choices")
        if option_type == "str" and isinstance(choices, list) and choices:
            selected[key] = st.selectbox(
                label,
                options=[str(choice) for choice in choices],
                index=max(0, [str(choice) for choice in choices].index(default) if default in choices else 0),
                key=widget_key,
                help=description or None,
            )
            continue
        selected[key] = st.text_input(
            label,
            value=str(st.session_state.get(widget_key, default)),
            key=widget_key,
            help=description or None,
        )
    return selected


def _render_case_detail(case: AgenticCaseInfo) -> None:
    st.write(case.description)
    cols = st.columns(3)
    cols[0].metric("Turnos", len(case.turns))
    cols[1].metric("Escenario", case.experiment_case_id)
    cols[2].metric("Audit reads", "sí" if case.audit_reads else "no")
    st.caption(f"Tags: {', '.join(case.tags) or '—'}")
    st.caption(f"Requisitos: {', '.join(case.requirements) or '—'}")
    _render_editable_prompts(case)


def _prompt_widget_key(case_id: str, index: int) -> str:
    """Return a stable Streamlit widget key for one case turn prompt."""
    return f"testing_prompt__{case_id}__{index}"


def _ensure_prompt_defaults(case: AgenticCaseInfo) -> None:
    """Seed editable prompts from the case catalog when missing or stale."""
    fingerprint_key = f"testing_prompt_fingerprint__{case.id}"
    fingerprint = (len(case.turns), case.turns)
    if st.session_state.get(fingerprint_key) != fingerprint:
        for index, prompt in enumerate(case.turns):
            st.session_state[_prompt_widget_key(case.id, index)] = prompt
        st.session_state[fingerprint_key] = fingerprint
        return
    for index, prompt in enumerate(case.turns):
        key = _prompt_widget_key(case.id, index)
        if key not in st.session_state:
            st.session_state[key] = prompt


def _edited_prompts(case: AgenticCaseInfo) -> tuple[str, ...]:
    """Return the current editable prompts for ``case``."""
    _ensure_prompt_defaults(case)
    return tuple(str(st.session_state[_prompt_widget_key(case.id, index)]) for index in range(len(case.turns)))


def _reset_edited_prompts(case: AgenticCaseInfo) -> None:
    """Restore editable prompts to the catalog defaults for ``case``."""
    for index, prompt in enumerate(case.turns):
        st.session_state[_prompt_widget_key(case.id, index)] = prompt
    st.session_state[f"testing_prompt_fingerprint__{case.id}"] = (len(case.turns), case.turns)


def _render_editable_prompts(case: AgenticCaseInfo) -> None:
    """Render per-turn prompt editors that persist in Streamlit session state."""
    _ensure_prompt_defaults(case)
    with st.expander("Prompts del caso", expanded=True):
        st.caption("Edita los prompts y lanza el caso sin reiniciar la app.")
        for index in range(len(case.turns)):
            st.text_area(
                f"Turno {index + 1}",
                key=_prompt_widget_key(case.id, index),
                height=140,
            )
        if st.button(
            "Restablecer prompts",
            icon=":material/restart_alt:",
            key=f"testing_reset_prompts__{case.id}",
        ):
            _reset_edited_prompts(case)
            st.rerun()


def _render_provider_status() -> tuple[bool, str | None]:
    """Render catalog YAML picker; return ``(ready, selected_catalog_path)``."""
    catalog_paths = list_packaged_llm_catalog_paths()
    if not catalog_paths:
        st.error("No hay catálogos LLM (`custom.yaml` / `default.yaml`) en `jarl/agentic/llm/profiles/`.")
        return False, None

    path_by_label = {path.stem: path for path in catalog_paths}
    labels = list(path_by_label)
    resolved = resolve_llm_yaml_path()
    default_label = labels[0]
    if resolved is not None:
        for label, path in path_by_label.items():
            if path.resolve() == resolved.resolve():
                default_label = label
                break

    selected_label = st.selectbox(
        "Catálogo LLM",
        options=labels,
        index=labels.index(default_label),
        key="testing_llm_catalog",
        help=("Catálogo YAML completo (`default` / `custom`). Respeta assignments de main y subagents."),
    )
    selected_path = path_by_label[selected_label]
    try:
        catalog = load_llm_catalog(path=selected_path)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        st.error(f"Configuración LLM inválida: {exc}")
        return False, None

    main = catalog.resolve(MAIN_COMPONENT_ID)
    sub_default = catalog.resolve("subagents/other")
    metrics = catalog.resolve("subagents/metrics_analysis")
    st.info(
        f"Catálogo: **{selected_label}** · "
        f"main→`{main.profile_name}` ({main.settings.provider}/{main.settings.model}) · "
        f"subagents/*→`{sub_default.profile_name}` · "
        f"metrics_analysis→`{metrics.profile_name}`"
    )
    return True, str(selected_path)


def _latest_experiment_dir() -> Path | None:
    active = active_agentic_case_run()
    if active is not None:
        return active.experiment_dir
    raw = st.session_state.get(LAST_AGENTIC_CASE_DIR_KEY)
    return Path(raw) if raw else None


def _case_finished() -> bool:
    """Return whether the latest subprocess finished (exit code recorded)."""
    return active_agentic_case_run() is None and st.session_state.get(LAST_AGENTIC_CASE_EXIT_CODE_KEY) is not None


def _render_open_experiment_button(
    experiment_dir: Path,
    *,
    key: str = "testing_open_experiment",
    require_finished: bool = True,
) -> None:
    """Show tree navigation when the experiment manifest exists."""
    if require_finished and not _case_finished():
        return
    if not (experiment_dir / "experiment.json").is_file():
        return
    if st.button(
        "Abrir experimento en Árbol",
        icon=":material/account_tree:",
        key=key,
    ):
        _ensure_cases_workspace()
        set_experiment_dir(experiment_dir)
        st.switch_page(nav_page_path("arbol"))


def _render_case_result_fragment() -> None:
    watching_result = bool(st.session_state.get(AGENTIC_RESULT_WATCH_KEY))
    interval = DEFAULT_RUN_STATUS_REFRESH_SECONDS if watching_result else None

    @st.fragment(run_every=interval)
    def _poll_result() -> None:
        experiment_dir = _latest_experiment_dir()
        if experiment_dir is None:
            st.caption("Ejecuta un caso para ver su reporte.")
            return

        _render_open_experiment_button(experiment_dir)

        try:
            result = load_agentic_case_result(experiment_dir)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            st.error(f"No se pudo leer el resultado: {exc}")
            return
        if result is None:
            if _case_finished():
                st.caption("Caso terminado. Esperando ValidationReport…")
            else:
                st.caption("Esperando ValidationReport…")
            return

        st.session_state[AGENTIC_RESULT_WATCH_KEY] = False
        _render_case_result_body(result)

    _poll_result()
