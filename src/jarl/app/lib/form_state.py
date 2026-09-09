"""Persisted form defaults shared across Runner Lab pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jarl.training.config import RLRunConfig
from jarl.training.presets import (
    DEFAULT_CUSTOM_BASE,
    DEFAULT_SAVE_MODEL,
    AlgorithmChoice,
    CustomBaseKind,
    PresetKind,
    RunFormValues,
    custom_base_form_values,
    default_form_values,
    form_values_from_run_config,
    recommended_video_frequency,
)

FORM_DEFAULTS_REVISION = 6
PREVIOUS_FORM_DEFAULTS_REVISION = 5
LAB_PREFIX = f"lab_form_v{FORM_DEFAULTS_REVISION}_"
REVISION_KEY = f"{LAB_PREFIX}defaults_revision"
RESET_REQUEST_KEY = f"{LAB_PREFIX}reset_requested"
CUSTOM_BASE_REQUEST_KEY = f"{LAB_PREFIX}custom_base_requested"
PARENT_RESET_NODE_KEY = f"{LAB_PREFIX}parent_reset_node"
NODE_CONFIG_RESET_NODE_KEY = f"{LAB_PREFIX}node_config_reset_node"
NODE_CONFIG_RESET_NOTICE_KEY = f"{LAB_PREFIX}node_config_reset_notice"
CONTINUATION_SOURCE_KEY = f"{LAB_PREFIX}continuation_source"
PENDING_CONTINUATION_SOURCE_KEY = f"{LAB_PREFIX}pending_continuation_source"
MIGRATION_NOTICE_KEY = f"{LAB_PREFIX}defaults_migrated"
CUSTOM_BASE_NOTICE_KEY = f"{LAB_PREFIX}custom_base_applied"

PERSISTED_FIELDS: tuple[str, ...] = (
    "algorithm",
    "preset",
    "custom_base",
    "env_id",
    "seed",
    "total_timesteps",
    "nr_envs",
    "nr_steps",
    "minibatch_size",
    "eval_frequency",
    "learning_rate",
    "obs_encoding_dim",
    "gru_hidden_dim",
    "track_wandb",
    "track_tensorboard",
    "wandb_project",
    "wandb_mode",
    "record_video",
    "video_frequency",
    "record_final_video",
    "gamma",
    "gae_lambda",
    "clip_range",
    "entropy_coef",
    "nr_epochs",
    "save_model",
    "evaluation_active",
    "checkpoint_save_interval_steps",
    "wandb_entity",
    "wandb_group",
    "wandb_tags",
    "experiment_name",
    "root_label",
)

CUSTOM_BASE_FIELDS: tuple[str, ...] = (
    "custom_base",
    "preset",
    "env_id",
    "seed",
    "total_timesteps",
    "nr_envs",
    "nr_steps",
    "minibatch_size",
    "eval_frequency",
    "learning_rate",
    "obs_encoding_dim",
    "gru_hidden_dim",
    "video_frequency",
    "gamma",
    "gae_lambda",
    "clip_range",
    "entropy_coef",
    "nr_epochs",
    "evaluation_active",
    "checkpoint_save_interval_steps",
)


def lab_key(field: str) -> str:
    """Return the session-state key for a persisted form field."""
    return f"{LAB_PREFIX}{field}"


def default_form_payload(
    *,
    algorithm: AlgorithmChoice = "ppo",
    custom_base: CustomBaseKind = DEFAULT_CUSTOM_BASE,
) -> dict[str, Any]:
    """Build the full persisted-field map for production-scale defaults."""
    defaults = custom_base_form_values(algorithm=algorithm, base=custom_base)
    payload = _values_to_dict(defaults)
    payload["custom_base"] = custom_base
    payload["video_frequency"] = recommended_video_frequency(
        total_timesteps=defaults.total_timesteps,
        eval_frequency=defaults.eval_frequency,
    )
    payload["record_final_video"] = True
    payload["save_model"] = DEFAULT_SAVE_MODEL
    payload["experiment_name"] = "navix_demo"
    payload["root_label"] = "baseline"
    return payload


def request_node_config_reset(node_id: str, notice: str) -> None:
    """Queue restoring the form from a node's resolved config."""
    import streamlit as st

    st.session_state[NODE_CONFIG_RESET_NODE_KEY] = node_id
    st.session_state[NODE_CONFIG_RESET_NOTICE_KEY] = notice


def request_parent_form_reset(node_id: str) -> None:
    """Queue restoring the form from a parent node's resolved config."""
    request_node_config_reset(node_id, notice="Formulario restaurado a la config resuelta del nodo origen.")


def apply_pending_node_config_reset(exp_dir: Path) -> None:
    """Apply a queued node-config reset before widgets are drawn."""
    import streamlit as st

    from jarl.app.lib.session import load_graph

    node_id = st.session_state.pop(NODE_CONFIG_RESET_NODE_KEY, None)
    if node_id is None:
        node_id = st.session_state.pop(PARENT_RESET_NODE_KEY, None)
    if node_id is None:
        return
    graph = load_graph(exp_dir)
    config = graph.resolve_config(graph.get_node(node_id))
    apply_form_from_run_config(config)
    st.session_state[CONTINUATION_SOURCE_KEY] = node_id
    st.session_state.setdefault(
        NODE_CONFIG_RESET_NOTICE_KEY,
        "Formulario restaurado a la config del nodo.",
    )


def apply_pending_parent_form_reset(exp_dir: Path) -> None:
    """Apply a queued node-config reset before widgets are drawn."""
    apply_pending_node_config_reset(exp_dir)


def pop_node_config_reset_notice() -> str | None:
    """Return and clear the success message for a node-config reset."""
    import streamlit as st

    notice = st.session_state.pop(NODE_CONFIG_RESET_NOTICE_KEY, None)
    if isinstance(notice, str) and notice:
        return notice
    return None


def apply_pending_continuation_sync(exp_dir: Path) -> None:
    """Apply a queued continuation-source sync before widgets are drawn."""
    import streamlit as st

    from jarl.app.lib.session import load_graph

    node_id = st.session_state.pop(PENDING_CONTINUATION_SOURCE_KEY, None)
    if node_id is None:
        return
    graph = load_graph(exp_dir)
    apply_form_from_run_config(graph.resolve_config(graph.get_node(str(node_id))))
    st.session_state[CONTINUATION_SOURCE_KEY] = str(node_id)


def apply_pending_form_mutations(exp_dir: Path | None) -> None:
    """Run queued form mutations before any ``lab_form_*`` widgets render."""
    apply_pending_form_reset()
    apply_pending_custom_base()
    if exp_dir is None:
        return
    apply_pending_node_config_reset(exp_dir)
    apply_pending_continuation_sync(exp_dir)


def sync_form_from_run_config_if_needed(*, source_key: str, config: RLRunConfig) -> None:
    """Sync the persisted form when the selected continuation node changes."""
    import streamlit as st

    if st.session_state.get(CONTINUATION_SOURCE_KEY) == source_key:
        return
    apply_form_from_run_config(config)
    st.session_state[CONTINUATION_SOURCE_KEY] = source_key


def apply_form_from_run_config(config: RLRunConfig) -> None:
    """Overwrite persisted form keys from a resolved run config."""
    values = form_values_from_run_config(config)
    payload = _values_to_dict(values)
    payload["preset"] = "custom"
    payload["custom_base"] = DEFAULT_CUSTOM_BASE
    payload["video_frequency"] = recommended_video_frequency(
        total_timesteps=values.total_timesteps,
        eval_frequency=values.eval_frequency,
    )
    payload["record_final_video"] = config.video.record_final_video
    for field, value in payload.items():
        set_persisted_field(field, value)
    import streamlit as st

    st.session_state[REVISION_KEY] = FORM_DEFAULTS_REVISION


def request_persisted_form_reset() -> None:
    """Queue a form reset for the next rerun before widgets are drawn."""
    import streamlit as st

    st.session_state[RESET_REQUEST_KEY] = True


def request_custom_base_apply() -> None:
    """Queue applying the selected custom base before widgets are drawn."""
    import streamlit as st

    st.session_state[CUSTOM_BASE_REQUEST_KEY] = st.session_state.get(
        lab_key("custom_base"),
        DEFAULT_CUSTOM_BASE,
    )


def apply_pending_form_reset() -> None:
    """Apply a queued reset. Must run before any ``lab_form_*`` widgets."""
    import streamlit as st

    if not st.session_state.pop(RESET_REQUEST_KEY, False):
        return
    algorithm = coerce_algorithm(st.session_state.get(lab_key("algorithm"), "ppo"))
    reset_persisted_form(algorithm=algorithm)


def apply_pending_custom_base() -> None:
    """Apply a queued custom base. Must run before widgets are drawn."""
    import streamlit as st

    raw_base = st.session_state.pop(CUSTOM_BASE_REQUEST_KEY, None)
    if raw_base is None:
        return
    custom_base = coerce_custom_base(raw_base)
    algorithm = coerce_algorithm(st.session_state.get(lab_key("algorithm"), "ppo"))
    apply_custom_base(algorithm=algorithm, custom_base=custom_base)
    st.session_state[CUSTOM_BASE_NOTICE_KEY] = custom_base


def reset_persisted_form(
    *,
    algorithm: AlgorithmChoice | None = None,
    custom_base: CustomBaseKind = DEFAULT_CUSTOM_BASE,
    notify: bool = False,
) -> None:
    """Overwrite all persisted form keys with production defaults."""
    import streamlit as st

    current_algorithm = algorithm or coerce_algorithm(st.session_state.get(lab_key("algorithm"), "ppo"))
    payload = default_form_payload(algorithm=current_algorithm, custom_base=custom_base)
    for field, value in payload.items():
        st.session_state[lab_key(field)] = value
    st.session_state[REVISION_KEY] = FORM_DEFAULTS_REVISION
    if notify:
        st.session_state[MIGRATION_NOTICE_KEY] = True


def init_persisted_form() -> None:
    """Seed or migrate shared form keys in ``st.session_state``."""
    import streamlit as st

    current_revision = st.session_state.get(REVISION_KEY)
    if current_revision != FORM_DEFAULTS_REVISION:
        if current_revision == PREVIOUS_FORM_DEFAULTS_REVISION:
            st.session_state[lab_key("save_model")] = DEFAULT_SAVE_MODEL
            st.session_state[REVISION_KEY] = FORM_DEFAULTS_REVISION
            return
        reset_persisted_form(notify=True)
        return

    defaults = default_form_values()
    payload = default_form_payload(algorithm=defaults.algorithm)
    _seed_missing_fields(payload)


def ensure_persisted_form_fields() -> None:
    """Seed missing fields without migrating or overwriting widget keys."""
    import streamlit as st

    algorithm = coerce_algorithm(st.session_state.get(lab_key("algorithm"), "ppo"))
    custom_base = coerce_custom_base(st.session_state.get(lab_key("custom_base"), DEFAULT_CUSTOM_BASE))
    payload = default_form_payload(algorithm=algorithm, custom_base=custom_base)
    _seed_missing_fields(payload)
    if REVISION_KEY not in st.session_state:
        st.session_state[REVISION_KEY] = FORM_DEFAULTS_REVISION


def apply_custom_base(*, algorithm: AlgorithmChoice, custom_base: CustomBaseKind) -> None:
    """Overwrite only fields governed by the selected custom base."""
    import streamlit as st

    payload = default_form_payload(algorithm=algorithm, custom_base=custom_base)
    payload["preset"] = "custom"
    for field in CUSTOM_BASE_FIELDS:
        st.session_state[lab_key(field)] = payload[field]


def _seed_missing_fields(payload: dict[str, Any]) -> None:
    """Set missing persisted keys only."""
    import streamlit as st

    for field, value in payload.items():
        key = lab_key(field)
        if key not in st.session_state:
            st.session_state[key] = value


def ensure_widget_key(field: str, value: Any) -> None:
    """Set a persisted widget key only when it is not already in session state."""
    import streamlit as st

    key = lab_key(field)
    if key not in st.session_state:
        st.session_state[key] = value


def pop_defaults_migration_notice() -> bool:
    """Return and clear whether form defaults were migrated this rerun."""
    import streamlit as st

    return bool(st.session_state.pop(MIGRATION_NOTICE_KEY, False))


def pop_custom_base_notice() -> CustomBaseKind | None:
    """Return and clear the custom base applied this rerun."""
    import streamlit as st

    raw_base = st.session_state.pop(CUSTOM_BASE_NOTICE_KEY, None)
    if raw_base is None:
        return None
    return coerce_custom_base(raw_base)


def load_run_form_values() -> tuple[RunFormValues, int, bool]:
    """Load persisted form values from session state."""
    import streamlit as st

    ensure_persisted_form_fields()
    values = RunFormValues(
        algorithm=st.session_state[lab_key("algorithm")],
        preset=st.session_state[lab_key("preset")],
        env_id=st.session_state[lab_key("env_id")],
        seed=int(st.session_state[lab_key("seed")]),
        total_timesteps=int(st.session_state[lab_key("total_timesteps")]),
        nr_envs=int(st.session_state[lab_key("nr_envs")]),
        nr_steps=int(st.session_state[lab_key("nr_steps")]),
        minibatch_size=int(st.session_state[lab_key("minibatch_size")]),
        eval_frequency=int(st.session_state[lab_key("eval_frequency")]),
        learning_rate=float(st.session_state[lab_key("learning_rate")]),
        obs_encoding_dim=int(st.session_state[lab_key("obs_encoding_dim")]),
        gru_hidden_dim=int(st.session_state[lab_key("gru_hidden_dim")]),
        track_wandb=bool(st.session_state[lab_key("track_wandb")]),
        track_tensorboard=bool(st.session_state[lab_key("track_tensorboard")]),
        wandb_project=str(st.session_state[lab_key("wandb_project")]),
        wandb_mode=str(st.session_state[lab_key("wandb_mode")]),
        record_video=bool(st.session_state[lab_key("record_video")]),
        gamma=float(st.session_state[lab_key("gamma")]),
        gae_lambda=float(st.session_state[lab_key("gae_lambda")]),
        clip_range=float(st.session_state[lab_key("clip_range")]),
        entropy_coef=float(st.session_state[lab_key("entropy_coef")]),
        nr_epochs=int(st.session_state[lab_key("nr_epochs")]),
        save_model=bool(st.session_state[lab_key("save_model")]),
        evaluation_active=bool(st.session_state[lab_key("evaluation_active")]),
        checkpoint_save_interval_steps=int(st.session_state[lab_key("checkpoint_save_interval_steps")]),
        wandb_entity=str(st.session_state[lab_key("wandb_entity")]),
        wandb_group=str(st.session_state[lab_key("wandb_group")]),
        wandb_tags=str(st.session_state[lab_key("wandb_tags")]),
        max_episode_steps=st.session_state.get(lab_key("max_episode_steps")),
    )
    video_frequency = int(st.session_state[lab_key("video_frequency")])
    record_final_video = bool(st.session_state[lab_key("record_final_video")])
    return values, video_frequency, record_final_video


def widget_value(field: str, fallback: Any) -> Any:
    """Read a persisted widget value from session state with fallback."""
    import streamlit as st

    init_persisted_form()
    return st.session_state.get(lab_key(field), fallback)


def set_persisted_field(field: str, value: Any) -> None:
    """Write a single persisted form field to session state."""
    import streamlit as st

    st.session_state[lab_key(field)] = value


def _values_to_dict(values: RunFormValues) -> dict[str, Any]:
    return {
        "algorithm": values.algorithm,
        "preset": values.preset,
        "env_id": values.env_id,
        "seed": values.seed,
        "total_timesteps": values.total_timesteps,
        "nr_envs": values.nr_envs,
        "nr_steps": values.nr_steps,
        "minibatch_size": values.minibatch_size,
        "eval_frequency": values.eval_frequency,
        "learning_rate": values.learning_rate,
        "obs_encoding_dim": values.obs_encoding_dim,
        "gru_hidden_dim": values.gru_hidden_dim,
        "track_wandb": values.track_wandb,
        "track_tensorboard": values.track_tensorboard,
        "wandb_project": values.wandb_project,
        "wandb_mode": values.wandb_mode,
        "record_video": values.record_video,
        "gamma": values.gamma,
        "gae_lambda": values.gae_lambda,
        "clip_range": values.clip_range,
        "entropy_coef": values.entropy_coef,
        "nr_epochs": values.nr_epochs,
        "save_model": values.save_model,
        "evaluation_active": values.evaluation_active,
        "checkpoint_save_interval_steps": values.checkpoint_save_interval_steps,
        "wandb_entity": values.wandb_entity,
        "wandb_group": values.wandb_group,
        "wandb_tags": values.wandb_tags,
        "max_episode_steps": values.max_episode_steps,
    }


def coerce_algorithm(value: object) -> AlgorithmChoice:
    """Coerce a session value to a supported algorithm choice."""
    if value in ("ppo", "ppo_gru"):
        return value  # type: ignore[return-value]
    return "ppo"


def coerce_preset(value: object) -> PresetKind:
    """Coerce a session value to a supported preset kind."""
    if value in ("fast", "custom"):
        return value  # type: ignore[return-value]
    return "custom"


def coerce_custom_base(value: object) -> CustomBaseKind:
    """Coerce a session value to a supported custom base."""
    if value in ("simple", "medium", "advanced"):
        return value  # type: ignore[return-value]
    return DEFAULT_CUSTOM_BASE
