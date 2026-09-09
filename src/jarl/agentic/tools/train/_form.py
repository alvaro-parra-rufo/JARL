"""Shared helpers for train tool form payloads and trainer resolution."""

from __future__ import annotations

from dataclasses import fields

from pydantic import BaseModel

from jarl.agentic.tools.context import ToolContext
from jarl.training.cli import resolve_trainer_for_node
from jarl.training.presets import (
    AlgorithmChoice,
    PresetKind,
    RunFormPayload,
    RunFormValues,
    default_form_values,
    preset_run_config,
    validate_run_form_values,
)
from jarl.training.trainer import Trainer

__all__ = [
    "RunFormPayloadRequest",
    "RunFormValuesRequest",
    "build_run_form_payload",
    "resolve_trainer",
]


class RunFormValuesRequest(BaseModel):
    """Partial run-form fields merged into preset defaults."""

    algorithm: AlgorithmChoice | None = None
    preset: PresetKind | None = None
    env_id: str | None = None
    seed: int | None = None
    total_timesteps: int | None = None
    nr_envs: int | None = None
    nr_steps: int | None = None
    minibatch_size: int | None = None
    eval_frequency: int | None = None
    learning_rate: float | None = None
    obs_encoding_dim: int | None = None
    gru_hidden_dim: int | None = None
    track_wandb: bool | None = None
    track_tensorboard: bool | None = None
    wandb_project: str | None = None
    wandb_mode: str | None = None
    record_video: bool | None = None
    gamma: float | None = None
    gae_lambda: float | None = None
    clip_range: float | None = None
    entropy_coef: float | None = None
    nr_epochs: int | None = None
    save_model: bool | None = None
    evaluation_active: bool | None = None
    checkpoint_save_interval_steps: int | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_tags: str | None = None
    max_episode_steps: int | None = None


class RunFormPayloadRequest(BaseModel):
    """Run-form payload aligned with ``RunFormPayload``."""

    values: RunFormValuesRequest | None = None
    video_frequency: int = 0
    record_final_video: bool = True
    reference_nr_envs: int | None = None


def build_run_form_payload(form: RunFormPayloadRequest | None) -> RunFormPayload | None:
    """Build ``RunFormPayload`` from a tool request, or ``None`` when absent."""
    if form is None:
        return None
    has_values = form.values is not None and any(value is not None for value in form.values.model_dump().values())
    has_video = form.video_frequency != 0 or not form.record_final_video
    has_reference = form.reference_nr_envs is not None
    if not (has_values or has_video or has_reference):
        return None
    values = _merge_form_values(form.values)
    errors = validate_run_form_values(values)
    if errors:
        msg = "; ".join(errors)
        raise ValueError(msg)
    return RunFormPayload(
        values=values,
        video_frequency=form.video_frequency,
        record_final_video=form.record_final_video,
        reference_nr_envs=form.reference_nr_envs,
    )


def resolve_trainer(ctx: ToolContext, *, node_id: str | None) -> Trainer:
    """Resolve the trainer for the target node."""
    return resolve_trainer_for_node(
        ctx.exp_dir,
        node_id=node_id,
        fallback_config=preset_run_config(),
    )


def _merge_form_values(request: RunFormValuesRequest | None) -> RunFormValues:
    preset: PresetKind = request.preset if request is not None and request.preset is not None else "fast"
    algorithm: AlgorithmChoice = request.algorithm if request is not None and request.algorithm is not None else "ppo"
    base = default_form_values(preset=preset, algorithm=algorithm)
    if request is None:
        return base
    overrides = {key: value for key, value in request.model_dump().items() if value is not None}
    if not overrides:
        return base
    merged = {field.name: getattr(base, field.name) for field in fields(RunFormValues)}
    merged.update(overrides)
    return RunFormValues(**merged)
