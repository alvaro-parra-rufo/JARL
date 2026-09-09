"""Page render helpers for the Runner Lab training module."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from jarl.app.lib.checkpoints_view import (
    available_checkpoint_aliases,
    checkpoint_ref,
    checkpoint_table,
    format_checkpoint_option,
    list_checkpoint_rows,
)
from jarl.app.lib.continue_flow import (
    CheckpointPick,
    ContinuationRequest,
    create_root_only,
    default_extend_label,
    default_fork_branch,
    overrides_from_payload,
    run_continuation,
)
from jarl.app.lib.form_state import (
    ensure_widget_key,
    init_persisted_form,
    lab_key,
    sync_form_from_run_config_if_needed,
)
from jarl.app.lib.forms import render_run_form
from jarl.app.lib.graph_ops import checkout_node
from jarl.app.lib.inspect_view import (
    can_extend_from_head,
    config_diff_dataframe,
    config_overrides_table,
    execution_attempts_table,
)
from jarl.app.lib.inspection_panel import render_node_reward_panel
from jarl.app.lib.layout import render_context_bar, select_node_workspace
from jarl.app.lib.live_feed import reset_log_feed
from jarl.app.lib.live_ui import (
    render_live_metrics_fragment,
    render_live_terminal_fragment,
)
from jarl.app.lib.runner_bridge import launch_create_and_train, launch_resume, launch_train_current
from jarl.app.lib.session import (
    active_run,
    bump_graph_revision,
    clear_last_created_node,
    last_created_node,
    list_node_summaries,
    load_graph,
    set_experiment_dir,
    set_last_created_node,
    workspace_root,
)
from jarl.app.lib.ui import (
    empty_state,
    key_value,
    path_block,
    render_parent_checkpoint_banner,
    render_status_badge,
)
from jarl.config import ConfigDiff
from jarl.experiments.io.checkpoints import CheckpointRef, checkpoint_ref_from_alias
from jarl.experiments.node import NodeStatus, NodeWorkspace
from jarl.experiments.summaries import config_highlights
from jarl.training.config import RLRunConfig
from jarl.training.presets import (
    RunFormPayload,
    form_values_to_run_config,
    validate_run_form_values,
)

__all__ = [
    "render_form_validation_status",
    "render_live_training_monitor",
    "render_training_continue",
    "render_training_debug",
    "render_training_launch",
    "render_training_new",
    "training_launch_allowed",
]


def training_launch_allowed(
    *,
    validation_messages: list[str],
    config_ready: bool,
    extra_blockers: list[str] | None = None,
) -> bool:
    """Return whether the UI should enable launch actions."""
    if validation_messages or extra_blockers:
        return False
    return config_ready


def render_form_validation_status(
    *,
    validation_messages: list[str],
    config_ready: bool,
    extra_blockers: list[str] | None = None,
) -> bool:
    """Render a compact validation strip and return whether launch is allowed."""
    blockers = list(validation_messages)
    if extra_blockers:
        blockers.extend(extra_blockers)
    if blockers:
        st.error("Corrige la configuración antes de lanzar:")
        for message in blockers:
            st.markdown(f"- {message}")
        return False
    if config_ready:
        st.success("Configuración válida — listo para lanzar.")
        return True
    st.warning("Completa el formulario para validar la config.")
    return False


def render_live_training_monitor(
    *,
    exp_dir: Path,
    workspace: NodeWorkspace,
    target_node: str,
) -> None:
    """Render prominent live KPI and log terminals for training."""
    train_buffer_key = f"jarl_terminal_train_{target_node}"
    st.subheader("Monitor en vivo")
    if active_run() is not None or workspace.metrics_jsonl_path.is_file():
        render_live_metrics_fragment(
            workspace.metrics_jsonl_path,
            fragment_key=f"entrenar_metrics_{target_node}",
        )
    else:
        empty_state("Sin métricas en vivo", "Lanza un entrenamiento para ver KPIs aquí.")

    log_tabs = st.tabs(("train.log", "runner_lab.log"))
    with log_tabs[0]:
        render_live_terminal_fragment(
            workspace.log_path,
            buffer_key=train_buffer_key,
            empty_message="Esperando train.log…",
        )
    with log_tabs[1]:
        render_live_terminal_fragment(
            exp_dir / "runner_lab.log",
            buffer_key="jarl_terminal_runner_lab",
            empty_message="Esperando runner_lab.log…",
        )


def reset_train_log_feed(workspace: NodeWorkspace, target_node: str) -> None:
    """Reset tail cursors for the active node log."""
    reset_log_feed(workspace.log_path, f"jarl_terminal_train_{target_node}")


def render_training_new() -> None:
    """Create a new experiment and optionally launch the first training run."""
    init_persisted_form()
    workspace = workspace_root()
    col_name, col_label = st.columns(2)
    with col_name:
        ensure_widget_key("experiment_name", "navix_demo")
        name = st.text_input("Nombre", key=lab_key("experiment_name"))
    with col_label:
        ensure_widget_key("root_label", "baseline")
        label = st.text_input("Etiqueta root", key=lab_key("root_label"))

    with st.expander("Configuración del experimento", expanded=True):
        payload = render_run_form()
    validation_messages = validate_run_form_values(payload.values)
    config = _config_from_payload(payload, validation_messages)

    exp_dir = workspace / name.strip()
    st.subheader("Destino")
    path_block(exp_dir)

    directory_blocked = exp_dir.exists() and any(exp_dir.iterdir())
    name_blocked = not name.strip()
    extra_blockers: list[str] = []
    if directory_blocked:
        extra_blockers.append("El directorio ya existe y no está vacío.")
    if name_blocked:
        extra_blockers.append("Nombre vacío.")

    can_launch = render_form_validation_status(
        validation_messages=validation_messages,
        config_ready=config is not None,
        extra_blockers=extra_blockers,
    )

    if config is not None:
        preview, raw = st.tabs(("Resumen", "config.json"))
        with preview:
            key_value(config_highlights(config))
        with raw:
            st.json(config.model_dump(), expanded=False)

    col_train, col_prepare = st.columns(2)
    with col_train:
        if (
            st.button(
                "Crear y entrenar",
                type="primary",
                icon=":material/play_arrow:",
                disabled=not can_launch,
            )
            and config is not None
        ):
            launch_create_and_train(experiment_dir=exp_dir, config=config, label=label)
            set_experiment_dir(exp_dir)
            st.success("Entrenamiento lanzado. Abre la pestaña **Lanzar** para el monitor en vivo.")
    with col_prepare:
        if (
            st.button(
                "Solo preparar root",
                icon=":material/inventory_2:",
                disabled=not can_launch,
            )
            and config is not None
        ):
            node_id = create_root_only(experiment_dir=exp_dir, config=config, label=label)
            set_experiment_dir(exp_dir)
            set_last_created_node(node_id)
            st.success(f"Root preparado: {node_id}. Usa **Continuar** o **Lanzar**.")


def render_training_launch(exp_dir: Path) -> None:
    """Train or resume the selected experiment node."""
    graph = load_graph(exp_dir)
    path_block(exp_dir)

    target_workspace = select_node_workspace(
        graph,
        exp_dir,
        session_key="entrenar_node",
        label="Nodo objetivo",
    )
    target_node = target_workspace.id
    render_context_bar(target_workspace)
    target_config = graph.resolve_config(target_workspace)
    sync_form_from_run_config_if_needed(source_key=target_node, config=target_config)
    render_parent_checkpoint_banner(target_workspace)

    can_train = target_workspace.status in {NodeStatus.CREATED, NodeStatus.PREPARED}
    can_resume = target_workspace.status in {NodeStatus.FAILED, NodeStatus.INTERRUPTED}

    render_live_training_monitor(
        exp_dir=exp_dir,
        workspace=target_workspace,
        target_node=target_node,
    )

    node_config_resets: list[tuple[str, str, str]] = [
        (
            target_node,
            "Config del nodo",
            "Formulario restaurado a la config resuelta del nodo seleccionado.",
        ),
    ]
    parent_id = target_workspace.node_metadata.parent_id
    if parent_id is not None:
        node_config_resets.append(
            (
                parent_id,
                "Config del padre",
                "Formulario restaurado a la config resuelta del nodo padre.",
            )
        )

    with st.expander("Configuración de entrenamiento", expanded=not active_run()):
        payload = render_run_form(
            include_environment=False,
            environment_nr_envs=target_config.environment.nr_envs,
            lock_algorithm=True,
            node_config_resets=tuple(node_config_resets),
        )
    validation_messages = validate_run_form_values(
        payload.values,
        environment_nr_envs=target_config.environment.nr_envs,
    )
    config = _config_from_payload(payload, validation_messages)

    status_blockers: list[str] = []
    if target_workspace.status == NodeStatus.COMPLETED:
        status_blockers.append("Nodo completado: crea un fork en Árbol o Continuar.")
    elif not can_train and not can_resume:
        status_blockers.append(f"Estado `{target_workspace.status.value}` no admite train/resume.")

    can_launch = render_form_validation_status(
        validation_messages=validation_messages,
        config_ready=config is not None,
        extra_blockers=status_blockers,
    )

    if config is not None:
        with st.expander("Config enviada al subprocess", expanded=False):
            key_value(config_highlights(config))
            st.json(config.model_dump(), expanded=False)

    col1, col2, col3 = st.columns(3)
    with col1:
        if (
            st.button(
                "Entrenar nodo",
                type="primary",
                icon=":material/play_arrow:",
                disabled=not can_launch or not can_train,
            )
            and config is not None
        ):
            launch_train_current(experiment_dir=exp_dir, config=config, node_id=target_node)
            reset_train_log_feed(target_workspace, target_node)
            st.success("Entrenamiento lanzado.")
    with col2:
        if (
            st.button(
                "Reanudar nodo",
                icon=":material/replay:",
                disabled=not can_launch or not can_resume,
            )
            and config is not None
        ):
            launch_resume(experiment_dir=exp_dir, node_id=target_node, config=config)
            reset_train_log_feed(target_workspace, target_node)
            st.success("Resume lanzado.")
    with col3:
        if st.button("Refrescar", icon=":material/refresh:"):
            st.rerun()


def render_training_continue(exp_dir: Path) -> None:  # noqa: PLR0912
    """Continuation wizard for extend, fork, and checkpoint flows."""
    graph = load_graph(exp_dir)
    path_block(exp_dir)

    summaries = list_node_summaries(exp_dir, graph=graph)
    node_ids = [node_id for node_id, _ in summaries]
    status_by_node = dict(summaries)

    st.subheader("1. Origen")
    source_node = st.selectbox(
        "Nodo origen",
        options=node_ids,
        index=node_ids.index(graph.current_node.id),
        format_func=lambda node_id: f"{node_id} · {status_by_node[node_id]}",
        key="continuar_source_node",
    )
    parent_workspace = graph.get_node(source_node)
    parent_config = graph.resolve_config(parent_workspace)
    sync_form_from_run_config_if_needed(source_key=source_node, config=parent_config)
    render_status_badge(parent_workspace.status.value)
    key_value(
        {
            "branch": parent_workspace.branch,
            "label": parent_workspace.node_metadata.label,
            "status": parent_workspace.status.value,
            "is_branch_head": can_extend_from_head(graph, source_node),
            "env_id": parent_config.environment.env_id,
            "nr_envs": parent_config.environment.nr_envs,
            "algorithm": parent_config.algorithm.name,
        }
    )

    st.subheader("2. Modo de continuación")
    kind = st.radio(
        "Acción",
        options=("extend", "fork"),
        format_func=lambda value: "Extend (misma rama)" if value == "extend" else "Fork (nueva rama)",
        horizontal=True,
        key="continuar_kind",
    )
    if kind == "extend" and not can_extend_from_head(graph, source_node):
        st.warning("Extend solo está disponible cuando el nodo origen es el **head** de su rama.")

    branch = ""
    if kind == "fork":
        branch = st.text_input("Nueva rama", value=default_fork_branch(parent_workspace), key="continuar_branch")
        branch_exists = branch in graph.branch_heads
        if branch_exists:
            st.error("La rama ya existe.")
    else:
        branch_exists = False
        st.caption(f"Rama objetivo: **{parent_workspace.branch}** (head avanzará)")

    label = st.text_input("Etiqueta del hijo", value=default_extend_label(parent_workspace), key="continuar_label")

    st.subheader("3. Checkpoint del padre")
    ckpt_rows = list_checkpoint_rows(parent_workspace)
    use_checkpoint = st.toggle(
        "Restaurar pesos desde checkpoint del padre",
        value=bool(ckpt_rows),
        key="continuar_use_checkpoint",
    )
    checkpoint_ref_value: CheckpointRef | None = None
    checkpoint_step: int | None = None

    if use_checkpoint:
        if not ckpt_rows:
            st.warning("El nodo origen no tiene checkpoints guardados.")
            use_checkpoint = False
        else:
            st.dataframe(checkpoint_table(ckpt_rows), use_container_width=True, hide_index=True)
            alias_options = ["step", *available_checkpoint_aliases(parent_workspace)]
            default_alias_index = alias_options.index("latest") if "latest" in alias_options else 0
            pick_mode = st.radio(
                "Selección",
                options=alias_options,
                index=default_alias_index,
                format_func=lambda value: value if value != "step" else "step concreto",
                horizontal=True,
                key="continuar_ckpt_mode",
            )
            if pick_mode == "step":
                selected_row = st.selectbox(
                    "Checkpoint",
                    options=ckpt_rows,
                    format_func=format_checkpoint_option,
                    key="continuar_ckpt_row",
                )
                checkpoint_step = selected_row.checkpoint_step
                checkpoint_ref_value = checkpoint_ref(parent_workspace, checkpoint_step)
            else:
                checkpoint_ref_value = checkpoint_ref_from_alias(parent_workspace, pick_mode)
                if checkpoint_ref_value is None:
                    st.error(f"No hay checkpoint resuelto para alias `{pick_mode}`.")
                else:
                    checkpoint_step = checkpoint_ref_value.checkpoint_step

    prepare_only = st.toggle("Solo preparar (sin entrenar)", value=False, key="continuar_prepare_only")

    st.subheader("4. Overrides de entrenamiento")
    with st.expander("Formulario de overrides", expanded=True):
        payload = render_run_form(
            include_map_override=True,
            environment_nr_envs=parent_config.environment.nr_envs,
            lock_algorithm=True,
            node_config_resets=(
                (
                    source_node,
                    "Config del origen",
                    "Formulario restaurado a la config resuelta del nodo origen.",
                ),
            ),
        )
    validation_messages = validate_run_form_values(
        payload.values,
        environment_nr_envs=parent_config.environment.nr_envs,
    )
    config = None
    overrides: dict[str, object] | None = None
    diff = ConfigDiff(added={}, removed={}, changed={})
    if not validation_messages:
        try:
            config = form_values_to_run_config(payload)
            overrides = overrides_from_payload(parent_config, payload)
            child_preview = parent_config.apply_overrides(overrides)
            diff = parent_config.diff(child_preview, flatten=True)
        except Exception as exc:
            validation_messages = [str(exc)]

    render_form_validation_status(
        validation_messages=validation_messages,
        config_ready=overrides is not None,
    )

    if diff:
        st.caption("Cambios respecto a la config resuelta del padre:")
        diff_frame = config_diff_dataframe(diff)
        if diff_frame.empty:
            st.info("Sin overrides respecto al padre; el hijo heredará la misma config de entrenamiento.")
        else:
            st.dataframe(diff_frame, use_container_width=True, hide_index=True)

    blocked = (
        bool(validation_messages)
        or (kind == "extend" and not can_extend_from_head(graph, source_node))
        or (kind == "fork" and (not branch or branch_exists))
        or (use_checkpoint and checkpoint_ref_value is None)
    )

    if st.button(
        "Crear nodo hijo",
        type="primary",
        icon=":material/fork_right:",
        disabled=blocked or overrides is None,
        key="continuar_create_child",
    ):
        request = ContinuationRequest(
            kind=kind,  # type: ignore[arg-type]
            from_node=source_node,
            branch=branch if kind == "fork" else parent_workspace.branch,
            label=label,
            prepare=prepare_only,
            checkpoint_pick=CheckpointPick.STEP if use_checkpoint else CheckpointPick.NONE,
            checkpoint_step=checkpoint_step,
            config_overrides=overrides,
        )
        result = run_continuation(exp_dir, request=request, checkpoint_ref=checkpoint_ref_value)
        bump_graph_revision()
        set_last_created_node(result.child_id)
        checkout_node(exp_dir, result.child_id)
        st.success(f"Nodo creado: {result.child_id}")
        if result.parent_checkpoint_step is not None:
            st.info(
                f"Al entrenar restaurará pesos del checkpoint {result.parent_checkpoint_step} del padre {source_node}."
            )
        st.rerun()

    created = last_created_node()
    if created is not None:
        st.divider()
        st.subheader("Entrenar ahora")
        child_workspace = graph.get_node(created)
        render_status_badge(child_workspace.status.value)
        render_parent_checkpoint_banner(child_workspace)
        can_train = child_workspace.status in {NodeStatus.CREATED, NodeStatus.PREPARED}
        if not can_train:
            st.caption("El nodo ya no está en estado entrenable desde aquí.")
        elif config is not None and st.button("Entrenar ahora", type="primary", icon=":material/play_arrow:"):
            launch_train_current(experiment_dir=exp_dir, config=config, node_id=created)
            clear_last_created_node()
            st.success("Entrenamiento lanzado. Revisa **Lanzar** para el monitor.")


def render_training_debug(exp_dir: Path) -> None:
    """Debug view with config diff and execution attempts for one node."""
    graph = load_graph(exp_dir)
    summaries = list_node_summaries(exp_dir, graph=graph)
    node_ids = [node_id for node_id, _ in summaries]
    if not node_ids:
        empty_state("Sin nodos", "Crea un experimento primero.")
        return

    inspect_node = st.selectbox(
        "Nodo",
        options=node_ids,
        index=node_ids.index(graph.current_node.id) if graph.current_node.id in node_ids else 0,
        key="debug_inspect_node",
    )
    workspace = graph.get_node(inspect_node)
    render_context_bar(workspace)
    render_parent_checkpoint_banner(workspace)

    parent_id = workspace.node_metadata.parent_id
    diff_tab, attempts_tab, overrides_tab, reward_tab = st.tabs(
        ("Diff vs padre", "Execution attempts", "Overrides", "Recompensa")
    )
    with diff_tab:
        if parent_id is None:
            st.caption("El nodo root no tiene padre para comparar.")
        else:
            diff = graph.get_config_diff(parent_id, inspect_node)
            diff_df = config_diff_dataframe(diff)
            if diff_df.empty:
                st.info("Sin diferencias de config respecto al padre.")
            else:
                st.dataframe(diff_df, use_container_width=True, hide_index=True)

    with attempts_tab:
        attempts_df = execution_attempts_table(workspace)
        if attempts_df.empty:
            empty_state("Sin intentos", "Aún no hay execution_attempts.json.")
        else:
            st.dataframe(attempts_df, use_container_width=True, hide_index=True)

    with overrides_tab:
        overrides_df = config_overrides_table(workspace)
        if overrides_df.empty:
            empty_state("Sin overrides", "El nodo no guardó config_overrides.json.")
        else:
            st.dataframe(overrides_df, use_container_width=True, hide_index=True)

    with reward_tab:
        render_node_reward_panel(graph, workspace)


def _config_from_payload(
    payload: RunFormPayload,
    validation_messages: list[str],
) -> RLRunConfig | None:
    if validation_messages:
        return None
    try:
        return form_values_to_run_config(payload)
    except Exception as exc:
        validation_messages.append(str(exc))
        return None
