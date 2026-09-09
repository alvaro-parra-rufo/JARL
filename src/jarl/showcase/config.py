"""Showcase pipeline configuration: scenario YAML, scales, and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from jarl.envs.navix.catalog import assert_transfer_compatible, get_contract, list_registered_maps
from jarl.training.config import RLRunConfig
from jarl.training.presets import AlgorithmChoice, production_run_config

SHOWCASE_DIR = Path(__file__).resolve().parent
DEFAULT_SCENARIO_PATH = SHOWCASE_DIR / "scenarios" / "default.yaml"
SCALES_DIR = SHOWCASE_DIR / "scenarios" / "scales"

NodeAction = Literal["root", "extend", "fork"]
ScaleName = Literal["smoke", "medium", "large"]
CHECKPOINT_ALIASES = frozenset({"best", "latest", "final"})
DEFAULT_SHOWCASE_WANDB_PROJECT = "jarl-showcase"

__all__ = [
    "CHECKPOINT_ALIASES",
    "DEFAULT_SHOWCASE_WANDB_PROJECT",
    "NodeAction",
    "NodeSpec",
    "ScaleConfig",
    "ScaleName",
    "ScenarioConfig",
    "ShowcaseAlgorithmChoice",
    "ShowcasePlan",
    "build_node_run_config",
    "load_scale",
    "load_scenario",
    "resolve_scale_path",
    "resolve_scenario_path",
    "validate_plan",
]

ShowcaseAlgorithmChoice = AlgorithmChoice


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """One logical node in a showcase scenario."""

    logical_name: str
    parent: str | None
    action: NodeAction
    branch: str
    label: str
    env_id: str
    checkpoint_alias: str | None
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Parsed showcase scenario."""

    name: str
    description: str
    algorithm: ShowcaseAlgorithmChoice
    nodes: tuple[NodeSpec, ...]


@dataclass(frozen=True, slots=True)
class ScaleConfig:
    """Training scale applied on top of a scenario."""

    name: str
    description: str
    wandb_mode: str
    timesteps_root: int
    timesteps_default: int
    timesteps_transfer: int
    nr_envs: int
    nr_steps: int
    minibatch_size: int
    nr_epochs: int
    eval_frequency: int
    checkpoint_save_interval_steps: int
    record_video: bool
    record_final_video: bool
    active_nodes: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class ShowcasePlan:
    """Merged scenario and scale ready for execution or dry-run."""

    experiment_name: str
    scenario: ScenarioConfig
    scale: ScaleConfig
    workspace_root: Path
    experiment_dir: Path
    nodes: tuple[NodeSpec, ...]
    algorithm: ShowcaseAlgorithmChoice
    wandb_online: bool
    wandb_project: str


def resolve_scenario_path(path: str | Path | None) -> Path:
    """Resolve a scenario YAML path."""
    if path is None:
        return DEFAULT_SCENARIO_PATH
    resolved = Path(path)
    if resolved.is_file():
        return resolved
    candidate = SHOWCASE_DIR / "scenarios" / resolved.name
    if candidate.is_file():
        return candidate
    msg = f"Scenario file not found: {path}"
    raise FileNotFoundError(msg)


def resolve_scale_path(scale: str | ScaleName) -> Path:
    """Resolve a scale YAML under ``scenarios/scales``."""
    path = SCALES_DIR / f"{scale}.yaml"
    if not path.is_file():
        msg = f"Unknown showcase scale: {scale!r}"
        raise FileNotFoundError(msg)
    return path


def load_scenario(path: str | Path | None = None) -> ScenarioConfig:
    """Load and parse a showcase scenario YAML."""
    scenario_path = resolve_scenario_path(path)
    payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Scenario YAML must be a mapping: {scenario_path}"
        raise ValueError(msg)
    nodes: list[NodeSpec] = []
    for raw in payload.get("nodes", []):
        if not isinstance(raw, dict):
            msg = "Each scenario node must be a mapping."
            raise ValueError(msg)
        nodes.append(
            NodeSpec(
                logical_name=str(raw["logical_name"]),
                parent=raw.get("parent"),
                action=raw["action"],
                branch=str(raw["branch"]),
                label=str(raw.get("label", raw["logical_name"])),
                env_id=str(raw["env_id"]),
                checkpoint_alias=raw.get("checkpoint_alias"),
                overrides=dict(raw.get("overrides") or {}),
            )
        )
    return ScenarioConfig(
        name=str(payload.get("name", scenario_path.stem)),
        description=str(payload.get("description", "")),
        algorithm=_parse_scenario_algorithm(payload.get("algorithm")),
        nodes=tuple(nodes),
    )


def load_scale(scale: str | ScaleName) -> ScaleConfig:
    """Load a scale YAML."""
    path = resolve_scale_path(scale)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Scale YAML must be a mapping: {path}"
        raise ValueError(msg)
    active = payload.get("active_nodes")
    active_nodes = tuple(str(item) for item in active) if active else None
    timesteps_default = int(payload["timesteps_default"])
    timesteps_transfer = int(payload.get("timesteps_transfer", timesteps_default))
    return ScaleConfig(
        name=str(payload.get("name", scale)),
        description=str(payload.get("description", "")),
        wandb_mode=str(payload.get("wandb_mode", "offline")),
        timesteps_root=int(payload["timesteps_root"]),
        timesteps_default=timesteps_default,
        timesteps_transfer=timesteps_transfer,
        nr_envs=int(payload["nr_envs"]),
        nr_steps=int(payload["nr_steps"]),
        minibatch_size=int(payload["minibatch_size"]),
        nr_epochs=int(payload.get("nr_epochs", 3)),
        eval_frequency=int(payload["eval_frequency"]),
        checkpoint_save_interval_steps=int(payload["checkpoint_save_interval_steps"]),
        record_video=bool(payload.get("record_video", False)),
        record_final_video=bool(payload.get("record_final_video", True)),
        active_nodes=active_nodes,
    )


def build_plan(
    *,
    experiment_name: str,
    workspace_root: Path,
    scenario_path: str | Path | None = None,
    scale_name: str = "medium",
    algorithm: ShowcaseAlgorithmChoice | None = None,
    wandb_online: bool = False,
    wandb_project: str = DEFAULT_SHOWCASE_WANDB_PROJECT,
) -> ShowcasePlan:
    """Merge scenario and scale into an executable plan."""
    scenario = load_scenario(scenario_path)
    scale = load_scale(scale_name)
    if scale.active_nodes is None:
        nodes = scenario.nodes
    else:
        active = set(scale.active_nodes)
        nodes = tuple(node for node in scenario.nodes if node.logical_name in active)
        missing = active - {node.logical_name for node in nodes}
        if missing:
            msg = f"Scale {scale.name!r} references unknown nodes: {sorted(missing)}"
            raise ValueError(msg)
    effective_wandb_online = wandb_online or scale.wandb_mode == "online"
    effective_algorithm = algorithm if algorithm is not None else scenario.algorithm
    return ShowcasePlan(
        experiment_name=experiment_name,
        scenario=scenario,
        scale=scale,
        workspace_root=workspace_root,
        experiment_dir=workspace_root / experiment_name,
        nodes=nodes,
        algorithm=effective_algorithm,
        wandb_online=effective_wandb_online,
        wandb_project=wandb_project,
    )


def validate_plan(plan: ShowcasePlan) -> list[str]:
    """Validate a showcase plan; return human-readable errors (empty if OK)."""
    errors: list[str] = []
    errors.extend(_validate_env_and_graph(plan))
    errors.extend(_validate_scale_batch(plan))
    errors.extend(_validate_node_batches(plan))
    return errors


def _validate_env_and_graph(plan: ShowcasePlan) -> list[str]:
    errors: list[str] = []
    registered = {contract.env_id for contract in list_registered_maps()}
    logical_names = {node.logical_name for node in plan.nodes}
    by_name = {node.logical_name: node for node in plan.nodes}

    for node in plan.nodes:
        if node.env_id not in registered:
            errors.append(f"{node.logical_name}: unknown env_id {node.env_id!r}")
        if node.action == "root":
            if node.parent is not None:
                errors.append(f"{node.logical_name}: root node must not have a parent")
            if node.checkpoint_alias is not None:
                errors.append(f"{node.logical_name}: root node must not use checkpoint_alias")
            continue
        if node.parent is None:
            errors.append(f"{node.logical_name}: non-root node requires parent")
        elif node.parent not in logical_names:
            errors.append(f"{node.logical_name}: parent {node.parent!r} not in active nodes")
        if node.checkpoint_alias is not None and node.checkpoint_alias not in CHECKPOINT_ALIASES:
            errors.append(
                f"{node.logical_name}: invalid checkpoint_alias {node.checkpoint_alias!r} "
                f"(expected one of {sorted(CHECKPOINT_ALIASES)})"
            )
        if node.parent is not None and node.parent in by_name:
            parent = by_name[node.parent]
            if parent.env_id != node.env_id:
                try:
                    assert_transfer_compatible(parent.env_id, node.env_id)
                except ValueError as exc:
                    errors.append(f"{node.logical_name}: {exc}")
                try:
                    get_contract(node.env_id)
                except KeyError as exc:
                    errors.append(f"{node.logical_name}: {exc}")

    errors.extend(_validate_topological_order(plan.nodes))
    return errors


def _validate_scale_batch(plan: ShowcasePlan) -> list[str]:
    scale = plan.scale
    rollout = scale.nr_envs * scale.nr_steps
    if rollout % scale.minibatch_size != 0:
        return [
            f"Scale {scale.name}: nr_envs * nr_steps ({rollout}) "
            f"must be divisible by minibatch_size ({scale.minibatch_size})"
        ]
    return []


def _validate_node_batches(plan: ShowcasePlan) -> list[str]:
    errors: list[str] = []
    for node in plan.nodes:
        config = build_node_run_config(plan, node)
        batch = config.environment.nr_envs * config.algorithm.nr_steps
        if batch % config.algorithm.minibatch_size != 0:
            errors.append(
                f"{node.logical_name}: nr_envs * nr_steps ({batch}) "
                f"not divisible by minibatch_size ({config.algorithm.minibatch_size})"
            )
    return errors


def build_node_run_config(plan: ShowcasePlan, node: NodeSpec) -> RLRunConfig:
    """Build the resolved run config for one logical node."""
    scale = plan.scale
    if node.action == "root":
        timesteps = scale.timesteps_root
    elif node.logical_name.startswith("transfer_"):
        timesteps = scale.timesteps_transfer
    else:
        timesteps = scale.timesteps_default
    config = production_run_config(algorithm=plan.algorithm).apply_overrides(
        {
            "environment.env_id": node.env_id,
            "environment.nr_envs": scale.nr_envs,
            "algorithm.total_timesteps": timesteps,
            "algorithm.nr_steps": scale.nr_steps,
            "algorithm.minibatch_size": scale.minibatch_size,
            "algorithm.nr_epochs": scale.nr_epochs,
            "algorithm.evaluation_and_save_frequency": scale.eval_frequency,
            "checkpoint.save_interval_steps": scale.checkpoint_save_interval_steps,
            "video.record_video": scale.record_video,
            "video.record_final_video": scale.record_final_video,
            "tracking.track_wandb": True,
            "tracking.track_tensorboard": True,
            "tracking.wandb_mode": "online" if plan.wandb_online else scale.wandb_mode,
            "tracking.wandb_project": plan.wandb_project,
            "tracking.wandb_group": plan.experiment_name,
            "tracking.wandb_tags": [node.logical_name, plan.scale.name],
        }
    )
    if node.overrides:
        config = config.apply_overrides(node.overrides)
    return config


def _parse_scenario_algorithm(raw: object) -> ShowcaseAlgorithmChoice:
    """Parse scenario ``algorithm`` field (``ppo`` or ``ppo_gru``)."""
    if raw is None:
        return "ppo"
    value = str(raw)
    if value not in ("ppo", "ppo_gru"):
        msg = f"Invalid scenario algorithm {value!r}; expected 'ppo' or 'ppo_gru'."
        raise ValueError(msg)
    return value  # type: ignore[return-value]


def _validate_topological_order(nodes: tuple[NodeSpec, ...]) -> list[str]:
    """Ensure nodes appear after their parents."""
    errors: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        if node.parent is not None and node.parent not in seen:
            errors.append(f"{node.logical_name}: parent {node.parent!r} must run before child")
        seen.add(node.logical_name)
    return errors
