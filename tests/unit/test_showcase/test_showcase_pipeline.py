"""Unit tests for the showcase pipeline (config, validation, dry-run)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.showcase.config import (
    NodeSpec,
    build_node_run_config,
    build_plan,
    load_scale,
    load_scenario,
    validate_plan,
)
from jarl.showcase.executor import ShowcaseExecutor, format_dry_run_plan
from jarl.training.presets import PPO_GRU_ALGORITHM_NAME


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    """Temporary workspace root for plan path assertions."""
    return tmp_path / "workspace"


def test_load_default_scenario_has_eleven_nodes() -> None:
    scenario = load_scenario()
    assert len(scenario.nodes) == 11
    assert scenario.algorithm == "ppo"
    assert scenario.nodes[0].logical_name == "baseline_empty5"
    assert scenario.nodes[0].action == "root"
    transfer_empty8 = next(node for node in scenario.nodes if node.logical_name == "transfer_empty8")
    assert transfer_empty8.action == "extend"
    assert transfer_empty8.branch == "transfer_spine"


def test_load_scales() -> None:
    smoke = load_scale("smoke")
    medium = load_scale("medium")
    assert smoke.active_nodes is not None
    assert len(smoke.active_nodes) == 3
    assert medium.active_nodes is None


def test_build_plan_smoke_subset(workspace_root: Path) -> None:
    plan = build_plan(
        experiment_name="test-showcase",
        workspace_root=workspace_root,
        scale_name="smoke",
    )
    assert len(plan.nodes) == 3
    assert plan.experiment_dir == workspace_root / "test-showcase"


def test_validate_plan_full_tree(workspace_root: Path) -> None:
    plan = build_plan(
        experiment_name="valid",
        workspace_root=workspace_root,
        scale_name="medium",
    )
    errors = validate_plan(plan)
    assert errors == []


def test_validate_plan_rejects_invalid_checkpoint_alias(workspace_root: Path) -> None:
    plan = build_plan(
        experiment_name="bad",
        workspace_root=workspace_root,
        scale_name="smoke",
    )
    node = plan.nodes[-1]
    broken_node = NodeSpec(
        logical_name=node.logical_name,
        parent=node.parent,
        action=node.action,
        branch=node.branch,
        label=node.label,
        env_id=node.env_id,
        checkpoint_alias="invalid",
        overrides=node.overrides,
    )
    broken_nodes = tuple(broken_node if item.logical_name == node.logical_name else item for item in plan.nodes)
    broken_plan = type(plan)(
        experiment_name=plan.experiment_name,
        scenario=plan.scenario,
        scale=plan.scale,
        workspace_root=plan.workspace_root,
        experiment_dir=plan.experiment_dir,
        nodes=broken_nodes,
        algorithm=plan.algorithm,
        wandb_online=plan.wandb_online,
        wandb_project=plan.wandb_project,
    )
    errors = validate_plan(broken_plan)
    assert any("checkpoint_alias" in error for error in errors)


def test_build_node_run_config_tags(workspace_root: Path) -> None:
    plan = build_plan(
        experiment_name="tagged",
        workspace_root=workspace_root,
        scale_name="smoke",
        wandb_project="tfm-jarl",
    )
    config = build_node_run_config(plan, plan.nodes[0])
    assert config.tracking.wandb_group == "tagged"
    assert config.tracking.wandb_project == "tfm-jarl"
    assert "baseline_empty5" in config.tracking.wandb_tags


def test_build_plan_algorithm_cli_overrides_scenario(workspace_root: Path) -> None:
    plan = build_plan(
        experiment_name="gru-run",
        workspace_root=workspace_root,
        scale_name="smoke",
        algorithm="ppo_gru",
    )
    assert plan.algorithm == "ppo_gru"
    config = build_node_run_config(plan, plan.nodes[0])
    assert config.algorithm.name == PPO_GRU_ALGORITHM_NAME


def test_build_plan_custom_workspace_root(workspace_root: Path) -> None:
    custom_root = workspace_root / "experiments"
    plan = build_plan(
        experiment_name="nested",
        workspace_root=custom_root,
        scale_name="smoke",
    )
    assert plan.experiment_dir == custom_root / "nested"


def test_build_node_run_config_uses_transfer_timesteps(workspace_root: Path) -> None:
    plan = build_plan(
        experiment_name="transfer-steps",
        workspace_root=workspace_root,
        scale_name="medium",
    )
    transfer = next(node for node in plan.nodes if node.logical_name == "transfer_empty6")
    refine = next(node for node in plan.nodes if node.logical_name == "main_extend_empty5")
    transfer_config = build_node_run_config(plan, transfer)
    refine_config = build_node_run_config(plan, refine)
    assert transfer_config.algorithm.total_timesteps == 200_000
    assert refine_config.algorithm.total_timesteps == 50_000


def test_dry_run_prints_commands(workspace_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan = build_plan(
        experiment_name="dry",
        workspace_root=workspace_root,
        scale_name="smoke",
    )
    executor = ShowcaseExecutor(plan)
    assert executor.run(dry_run=True) == 0
    captured = capsys.readouterr().out
    assert "baseline_empty5" in captured
    assert "dry-run" in captured.lower() or "dry" in captured.lower()


def test_format_dry_run_plan_lines(workspace_root: Path) -> None:
    plan = build_plan(
        experiment_name="lines",
        workspace_root=workspace_root,
        scale_name="smoke",
    )
    lines = format_dry_run_plan(plan)
    assert any("main_extend_empty5" in line for line in lines)


def test_existing_workspace_without_reuse_raises(tmp_path: Path) -> None:
    exp_dir = tmp_path / "existing"
    exp_dir.mkdir()
    (exp_dir / "experiment.json").write_text("{}", encoding="utf-8")
    plan = build_plan(
        experiment_name="existing",
        workspace_root=tmp_path,
        scale_name="smoke",
    )
    executor = ShowcaseExecutor(plan)
    with pytest.raises(FileExistsError):
        executor.run(dry_run=False)


def test_resolve_child_id_creates_fork_on_missing_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fork nodes on new branches must not call ``graph.head`` before the branch exists."""
    from jarl.experiments.graph import ExperimentGraph
    from jarl.experiments.io.checkpoints import CheckpointRef
    from jarl.training.config import RLRunConfig

    exp_dir = tmp_path / "fork-branch"
    exp_dir.mkdir()
    graph = ExperimentGraph(exp_dir, base_config=RLRunConfig())
    parent = graph.create_root(config=RLRunConfig(), branch="main", label="refine", prepare=True)
    graph.save()

    plan = build_plan(
        experiment_name="fork-branch",
        workspace_root=tmp_path,
        scale_name="medium",
    )
    node = next(item for item in plan.nodes if item.logical_name == "empty5_lr_low")
    executor = ShowcaseExecutor(plan)
    monkeypatch.setattr(
        executor,
        "_create_child_node",
        lambda *_args, **_kwargs: "lr_low_newchild_ab12cd34",
    )

    child_id = executor._resolve_child_id(
        node,
        parent.id,
        ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig),
        CheckpointRef(node_id=parent.id, checkpoint_step=1),
    )

    assert child_id == "lr_low_newchild_ab12cd34"
