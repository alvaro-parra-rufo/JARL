"""Streamlit form widgets for the Runner Lab."""

from __future__ import annotations

import streamlit as st

from jarl.app.lib.form_state import (
    apply_pending_custom_base,
    apply_pending_form_reset,
    coerce_algorithm,
    coerce_custom_base,
    coerce_preset,
    ensure_widget_key,
    init_persisted_form,
    lab_key,
    load_run_form_values,
    pop_custom_base_notice,
    pop_defaults_migration_notice,
    pop_node_config_reset_notice,
    request_custom_base_apply,
    request_node_config_reset,
    request_persisted_form_reset,
)
from jarl.app.lib.ui import key_value
from jarl.envs.navix.catalog import get_contract, list_registered_maps
from jarl.training.presets import (
    AlgorithmChoice,
    CustomBaseKind,
    PresetKind,
    RunFormPayload,
    RunFormValues,
    custom_base_form_values,
    default_form_values,
    preset_run_config,
    recommended_video_frequency,
)


def render_run_form(  # noqa: PLR0912
    *,
    include_environment: bool = True,
    include_map_override: bool = False,
    environment_nr_envs: int | None = None,
    lock_algorithm: bool = False,
    node_config_resets: tuple[tuple[str, str, str], ...] = (),
) -> RunFormPayload:
    """Render training widgets using persisted ``lab_form_*`` session keys.

    Each entry in ``node_config_resets`` is ``(node_id, button_label, success_notice)``.
    When empty, the global production defaults button is shown.

    Use ``include_map_override`` on fork/extend flows: only ``env_id`` is editable; ``nr_envs``
    and ``seed`` stay on the parent resolved config.
    """
    init_persisted_form()
    apply_pending_form_reset()
    apply_pending_custom_base()
    if pop_defaults_migration_notice():
        st.info("Defaults actualizados a producción: 5,000,000 timesteps, 64 envs, rollout 128.")
    reset_notice = pop_node_config_reset_notice()
    if reset_notice:
        st.success(reset_notice)
    custom_base_notice = pop_custom_base_notice()
    if custom_base_notice is not None:
        st.success(f"Base {_custom_base_label(custom_base_notice)} aplicada.")
    algorithm_choice = coerce_algorithm(st.session_state.get(lab_key("algorithm"), "ppo"))
    defaults = default_form_values(algorithm=algorithm_choice)
    header_col, reset_col = st.columns((4, 1))
    with header_col:
        algorithm_options: tuple[AlgorithmChoice, AlgorithmChoice] = ("ppo", "ppo_gru")
        ensure_widget_key("algorithm", algorithm_choice)
        if lock_algorithm:
            st.caption(
                "Algoritmo heredado del nodo origen: "
                f"{'PPO-GRU' if algorithm_choice == 'ppo_gru' else 'PPO feedforward'}"
            )
        else:
            algorithm_choice = st.radio(
                "Algoritmo",
                options=algorithm_options,
                index=algorithm_options.index(algorithm_choice),
                format_func=lambda value: "PPO feedforward" if value == "ppo" else "PPO-GRU",
                horizontal=True,
                key=lab_key("algorithm"),
            )
    with reset_col:
        st.write("")
        if node_config_resets:
            for node_id, label, notice in node_config_resets:
                st.button(
                    label,
                    icon=":material/restart_alt:",
                    use_container_width=True,
                    on_click=request_node_config_reset,
                    args=(node_id, notice),
                    key=f"{lab_key('reset')}_{node_id}",
                )
        else:
            st.button(
                "Defaults producción",
                icon=":material/restart_alt:",
                use_container_width=True,
                on_click=request_persisted_form_reset,
            )
    preset = coerce_preset(st.session_state.get(lab_key("preset"), "custom"))
    ensure_widget_key("preset", preset)
    preset = st.radio(
        "Modo",
        options=("fast", "custom"),
        format_func=lambda value: "Simple (smoke fijo)" if value == "fast" else "Personalizado",
        horizontal=True,
        key=lab_key("preset"),
    )
    editable = preset == "custom"
    defaults = default_form_values(algorithm=algorithm_choice)
    fast_config = preset_run_config(algorithm=algorithm_choice)

    if editable:
        custom_base_options: tuple[CustomBaseKind, CustomBaseKind, CustomBaseKind] = (
            "advanced",
            "medium",
            "simple",
        )
        custom_base = coerce_custom_base(st.session_state.get(lab_key("custom_base"), "advanced"))
        ensure_widget_key("custom_base", custom_base)
        base_col, apply_col = st.columns((3, 1))
        with base_col:
            custom_base = st.radio(
                "Base",
                options=custom_base_options,
                index=custom_base_options.index(custom_base),
                format_func=_custom_base_label,
                horizontal=True,
                key=lab_key("custom_base"),
            )
        with apply_col:
            st.write("")
            st.button(
                "Aplicar base",
                icon=":material/tune:",
                use_container_width=True,
                on_click=request_custom_base_apply,
            )
        base_preview = custom_base_form_values(algorithm=algorithm_choice, base=custom_base)
        key_value(
            {
                "timesteps base": base_preview.total_timesteps,
                "rollout base": f"{base_preview.nr_envs} envs x {base_preview.nr_steps} steps",
                "minibatch base": base_preview.minibatch_size,
                "eval base": base_preview.eval_frequency,
            }
        )
    else:
        key_value(
            {
                "modo": "simple fijo",
                "env_id": fast_config.environment.env_id,
                "timesteps": fast_config.algorithm.total_timesteps,
                "rollout": f"{fast_config.environment.nr_envs} envs x {fast_config.algorithm.nr_steps} steps",
                "minibatch": fast_config.algorithm.minibatch_size,
                "eval": fast_config.algorithm.evaluation_and_save_frequency,
            }
        )

    tab_names = ["Entrenamiento", "Vídeo", "Tracking", "Avanzado"]
    if include_environment:
        tab_names = ["Entorno", *tab_names]
    tabs = st.tabs(tuple(tab_names))
    tab_index = 0
    if include_environment:
        env_tab = tabs[tab_index]
        tab_index += 1
        with env_tab:
            if editable:
                navix_maps = list_registered_maps()
                env_options = [contract.env_id for contract in navix_maps] or [defaults.env_id]
                ensure_widget_key("env_id", defaults.env_id)
                current_env = str(st.session_state.get(lab_key("env_id"), defaults.env_id))
                if current_env not in env_options:
                    env_options = [current_env, *env_options]
                col1, col2 = st.columns(2)
                with col1:
                    env_id = st.selectbox(
                        "Mapa Navix",
                        options=env_options,
                        index=env_options.index(current_env),
                        key=lab_key("env_id"),
                    )
                    try:
                        contract = get_contract(env_id)
                        st.caption(f"transfer_group: `{contract.transfer_group}`")
                    except KeyError:
                        st.warning("Mapa no registrado en el catálogo Navix.")
                    ensure_widget_key("seed", defaults.seed)
                    seed = st.number_input(
                        "seed",
                        step=1,
                        key=lab_key("seed"),
                    )
                with col2:
                    ensure_widget_key("nr_envs", defaults.nr_envs)
                    nr_envs = st.number_input(
                        "nr_envs",
                        min_value=1,
                        step=1,
                        key=lab_key("nr_envs"),
                    )
            else:
                env_id = fast_config.environment.env_id
                seed = fast_config.environment.seed
                nr_envs = fast_config.environment.nr_envs
                key_value(
                    {
                        "Mapa Navix": env_id,
                        "seed": seed,
                        "nr_envs": nr_envs,
                        "max_episode_steps": fast_config.environment.max_episode_steps,
                    }
                )
    elif include_map_override:
        navix_maps = list_registered_maps()
        env_options = [contract.env_id for contract in navix_maps] or [defaults.env_id]
        ensure_widget_key("env_id", defaults.env_id)
        current_env = str(st.session_state.get(lab_key("env_id"), defaults.env_id))
        if current_env not in env_options:
            env_options = [current_env, *env_options]
        env_id = st.selectbox(
            "Mapa del hijo",
            options=env_options,
            index=env_options.index(current_env),
            key=lab_key("env_id"),
            help="Puedes cambiar de mapa si es transfer-compatible con el origen (mismo grupo Navix).",
        )
        try:
            contract = get_contract(env_id)
            st.caption(f"transfer_group: `{contract.transfer_group}`")
        except KeyError:
            st.warning("Mapa no registrado en el catálogo Navix.")
        effective_nr_envs = environment_nr_envs
        if effective_nr_envs is None:
            effective_nr_envs = int(st.session_state.get(lab_key("nr_envs"), defaults.nr_envs))
        st.caption(f"`nr_envs` y `seed` se heredan del nodo origen ({effective_nr_envs} envs).")
        nr_envs = effective_nr_envs
        seed = int(st.session_state.get(lab_key("seed"), defaults.seed))
    else:
        st.caption("Entorno fijo del nodo seleccionado; solo overrides de entrenamiento/tracking.")
        effective_nr_envs = environment_nr_envs
        if effective_nr_envs is None:
            effective_nr_envs = int(st.session_state.get(lab_key("nr_envs"), defaults.nr_envs))
        else:
            st.caption(f"Rollout batch usa `nr_envs={effective_nr_envs}` del nodo origen.")
        nr_envs = effective_nr_envs
        total_timesteps = int(st.session_state.get(lab_key("total_timesteps"), defaults.total_timesteps))
        eval_frequency = int(st.session_state.get(lab_key("eval_frequency"), defaults.eval_frequency))

    train_tab = tabs[tab_index]
    tab_index += 1
    video_tab = tabs[tab_index]
    tab_index += 1
    track_tab = tabs[tab_index]
    tab_index += 1
    advanced_tab = tabs[tab_index]

    with train_tab:
        if editable:
            col1, col2, col3 = st.columns(3)
            with col1:
                ensure_widget_key("total_timesteps", defaults.total_timesteps)
                total_timesteps = st.number_input(
                    "total_timesteps",
                    min_value=32,
                    step=32,
                    key=lab_key("total_timesteps"),
                )
                ensure_widget_key("nr_steps", defaults.nr_steps)
                nr_steps = st.number_input(
                    "nr_steps",
                    min_value=1,
                    step=1,
                    key=lab_key("nr_steps"),
                )
            with col2:
                ensure_widget_key("minibatch_size", defaults.minibatch_size)
                minibatch_size = st.number_input(
                    "minibatch_size",
                    min_value=1,
                    step=1,
                    key=lab_key("minibatch_size"),
                )
                ensure_widget_key("eval_frequency", defaults.eval_frequency)
                eval_frequency = st.number_input(
                    "evaluation_and_save_frequency",
                    min_value=-1,
                    step=1,
                    key=lab_key("eval_frequency"),
                )
            with col3:
                ensure_widget_key("learning_rate", defaults.learning_rate)
                learning_rate = st.number_input(
                    "learning_rate",
                    min_value=1e-6,
                    format="%.6f",
                    key=lab_key("learning_rate"),
                )
                st.metric("batch_size", int(nr_envs) * int(nr_steps))

            if algorithm_choice == "ppo_gru":
                gru_col1, gru_col2 = st.columns(2)
                with gru_col1:
                    ensure_widget_key("obs_encoding_dim", defaults.obs_encoding_dim)
                    obs_encoding_dim = st.number_input(
                        "obs_encoding_dim",
                        min_value=4,
                        step=4,
                        key=lab_key("obs_encoding_dim"),
                    )
                with gru_col2:
                    ensure_widget_key("gru_hidden_dim", defaults.gru_hidden_dim)
                    gru_hidden_dim = st.number_input(
                        "gru_hidden_dim",
                        min_value=4,
                        step=4,
                        key=lab_key("gru_hidden_dim"),
                    )
            else:
                obs_encoding_dim = defaults.obs_encoding_dim
                gru_hidden_dim = defaults.gru_hidden_dim
        else:
            total_timesteps = fast_config.algorithm.total_timesteps
            nr_steps = fast_config.algorithm.nr_steps
            minibatch_size = fast_config.algorithm.minibatch_size
            eval_frequency = fast_config.algorithm.evaluation_and_save_frequency
            learning_rate = fast_config.algorithm.learning_rate
            obs_encoding_dim = fast_config.algorithm.obs_encoding_dim
            gru_hidden_dim = fast_config.algorithm.gru_hidden_dim
            batch_size = fast_config.environment.nr_envs * fast_config.algorithm.nr_steps
            key_value(
                {
                    "total_timesteps": total_timesteps,
                    "nr_steps": nr_steps,
                    "minibatch_size": minibatch_size,
                    "learning_rate": learning_rate,
                    "eval_frequency": eval_frequency,
                    "batch_size": batch_size,
                }
            )

    with video_tab:
        ensure_widget_key("record_video", defaults.record_video)
        record_video = st.toggle(
            "Grabar vídeo Navix",
            key=lab_key("record_video"),
        )
        default_frequency = recommended_video_frequency(
            total_timesteps=int(total_timesteps),
            eval_frequency=int(eval_frequency),
        )
        if editable:
            ensure_widget_key("video_frequency", default_frequency)
            video_frequency = st.number_input(
                "video_frequency (env steps)",
                min_value=1,
                step=1,
                key=lab_key("video_frequency"),
                disabled=not record_video,
            )
        else:
            video_frequency = default_frequency
            key_value({"video_frequency": video_frequency, "video_source": "preset rápido"})
        ensure_widget_key("record_final_video", True)
        record_final_video = st.toggle(
            "Vídeo final",
            key=lab_key("record_final_video"),
            disabled=not record_video,
        )

    with track_tab:
        track_col1, track_col2 = st.columns(2)
        with track_col1:
            ensure_widget_key("track_wandb", defaults.track_wandb)
            track_wandb = st.toggle(
                "Weights & Biases",
                key=lab_key("track_wandb"),
            )
            ensure_widget_key("wandb_project", defaults.wandb_project)
            wandb_project = st.text_input(
                "wandb_project",
                key=lab_key("wandb_project"),
                disabled=not track_wandb,
            )
        with track_col2:
            ensure_widget_key("track_tensorboard", defaults.track_tensorboard)
            track_tensorboard = st.toggle(
                "TensorBoard",
                key=lab_key("track_tensorboard"),
            )
            wandb_modes = ("offline", "online", "disabled")
            ensure_widget_key("wandb_mode", defaults.wandb_mode)
            wandb_mode = st.selectbox(
                "wandb_mode",
                options=wandb_modes,
                key=lab_key("wandb_mode"),
                disabled=not track_wandb,
            )
        ensure_widget_key("wandb_entity", defaults.wandb_entity)
        wandb_entity = st.text_input("wandb_entity", key=lab_key("wandb_entity"), disabled=not track_wandb)
        ensure_widget_key("wandb_group", defaults.wandb_group)
        wandb_group = st.text_input("wandb_group", key=lab_key("wandb_group"), disabled=not track_wandb)
        ensure_widget_key("wandb_tags", defaults.wandb_tags)
        wandb_tags = st.text_input(
            "wandb_tags (coma)",
            key=lab_key("wandb_tags"),
            disabled=not track_wandb,
        )

    with advanced_tab:
        if editable:
            adv1, adv2, adv3 = st.columns(3)
            with adv1:
                ensure_widget_key("gamma", defaults.gamma)
                gamma = st.number_input("gamma", min_value=0.0, max_value=1.0, key=lab_key("gamma"))
                ensure_widget_key("gae_lambda", defaults.gae_lambda)
                gae_lambda = st.number_input(
                    "gae_lambda",
                    min_value=0.0,
                    max_value=1.0,
                    key=lab_key("gae_lambda"),
                )
                ensure_widget_key("clip_range", defaults.clip_range)
                clip_range = st.number_input(
                    "clip_range",
                    min_value=1e-6,
                    key=lab_key("clip_range"),
                )
            with adv2:
                ensure_widget_key("entropy_coef", defaults.entropy_coef)
                entropy_coef = st.number_input("entropy_coef", key=lab_key("entropy_coef"))
                ensure_widget_key("nr_epochs", defaults.nr_epochs)
                nr_epochs = st.number_input(
                    "nr_epochs",
                    min_value=1,
                    step=1,
                    key=lab_key("nr_epochs"),
                )
                ensure_widget_key("evaluation_active", defaults.evaluation_active)
                evaluation_active = st.toggle("evaluation_active", key=lab_key("evaluation_active"))
            with adv3:
                ensure_widget_key("save_model", defaults.save_model)
                save_model = st.toggle("save_model", key=lab_key("save_model"))
                ensure_widget_key("checkpoint_save_interval_steps", defaults.checkpoint_save_interval_steps)
                checkpoint_save_interval_steps = st.number_input(
                    "checkpoint.save_interval_steps",
                    min_value=1,
                    step=1,
                    key=lab_key("checkpoint_save_interval_steps"),
                )
        else:
            gamma = fast_config.algorithm.gamma
            gae_lambda = fast_config.algorithm.gae_lambda
            clip_range = fast_config.algorithm.clip_range
            entropy_coef = fast_config.algorithm.entropy_coef
            nr_epochs = fast_config.algorithm.nr_epochs
            evaluation_active = fast_config.algorithm.evaluation_active
            save_model = fast_config.runner.save_model
            checkpoint_save_interval_steps = fast_config.checkpoint.save_interval_steps
            key_value(
                {
                    "gamma": gamma,
                    "gae_lambda": gae_lambda,
                    "clip_range": clip_range,
                    "entropy_coef": entropy_coef,
                    "nr_epochs": nr_epochs,
                    "evaluation_active": evaluation_active,
                    "save_model": save_model,
                    "checkpoint": checkpoint_save_interval_steps,
                }
            )

    del (
        gamma,
        gae_lambda,
        clip_range,
        entropy_coef,
        nr_epochs,
        evaluation_active,
        save_model,
        checkpoint_save_interval_steps,
        wandb_entity,
        wandb_group,
        wandb_tags,
        nr_envs,
        total_timesteps,
        nr_steps,
        minibatch_size,
        eval_frequency,
        learning_rate,
        obs_encoding_dim,
        gru_hidden_dim,
        record_video,
        video_frequency,
        record_final_video,
        track_wandb,
        track_tensorboard,
        wandb_project,
        wandb_mode,
    )
    if include_environment or include_map_override:
        del env_id, seed

    return _payload_for_preset(
        preset,
        reference_nr_envs=environment_nr_envs if not include_environment else None,
    )


def _payload_for_preset(
    preset: PresetKind,
    *,
    reference_nr_envs: int | None = None,
) -> RunFormPayload:
    """Build a run payload without overwriting persisted widget values."""
    values, video_frequency, record_final_video = load_run_form_values()
    if preset != "fast":
        return RunFormPayload(
            values=values,
            video_frequency=video_frequency,
            record_final_video=record_final_video,
            reference_nr_envs=reference_nr_envs,
        )

    fast_config = preset_run_config(algorithm=values.algorithm)
    fast_algorithm = fast_config.algorithm
    fast_environment = fast_config.environment
    fast_values = RunFormValues(
        algorithm=values.algorithm,
        preset="fast",
        env_id=fast_environment.env_id,
        seed=fast_environment.seed,
        total_timesteps=fast_algorithm.total_timesteps,
        nr_envs=fast_environment.nr_envs,
        nr_steps=fast_algorithm.nr_steps,
        minibatch_size=fast_algorithm.minibatch_size,
        eval_frequency=fast_algorithm.evaluation_and_save_frequency,
        learning_rate=fast_algorithm.learning_rate,
        obs_encoding_dim=fast_algorithm.obs_encoding_dim,
        gru_hidden_dim=fast_algorithm.gru_hidden_dim,
        track_wandb=values.track_wandb,
        track_tensorboard=values.track_tensorboard,
        wandb_project=values.wandb_project,
        wandb_mode=values.wandb_mode,
        record_video=values.record_video,
        gamma=fast_algorithm.gamma,
        gae_lambda=fast_algorithm.gae_lambda,
        clip_range=fast_algorithm.clip_range,
        entropy_coef=fast_algorithm.entropy_coef,
        nr_epochs=fast_algorithm.nr_epochs,
        save_model=fast_config.runner.save_model,
        evaluation_active=fast_algorithm.evaluation_active,
        checkpoint_save_interval_steps=fast_config.checkpoint.save_interval_steps,
        wandb_entity=values.wandb_entity,
        wandb_group=values.wandb_group,
        wandb_tags=values.wandb_tags,
        max_episode_steps=fast_environment.max_episode_steps,
    )
    return RunFormPayload(
        values=fast_values,
        video_frequency=recommended_video_frequency(
            total_timesteps=fast_algorithm.total_timesteps,
            eval_frequency=fast_algorithm.evaluation_and_save_frequency,
        ),
        record_final_video=record_final_video,
        reference_nr_envs=reference_nr_envs,
    )


def _custom_base_label(value: CustomBaseKind) -> str:
    """Return the short UI label for a custom base profile."""
    labels: dict[CustomBaseKind, str] = {
        "advanced": "Avanzada",
        "medium": "Media",
        "simple": "Simple",
    }
    return labels[value]
