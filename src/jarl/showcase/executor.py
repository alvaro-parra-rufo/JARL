"""Sequential showcase executor: graph ops, training subprocesses, checkpoint videos."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import CheckpointRef, checkpoint_ref_from_alias
from jarl.experiments.manifest import MANIFEST_FILENAME
from jarl.experiments.node import NodeWorkspace
from jarl.showcase.checkpoint_video import render_checkpoint_video_subprocess
from jarl.showcase.config import NodeSpec, ShowcasePlan, build_node_run_config
from jarl.showcase.graph import extend_branch, fork_branch
from jarl.showcase.summary import SHOWCASE_RUN_FILENAME, save_showcase_run, write_showcase_summary
from jarl.training.config import RLRunConfig
from jarl.training.launch import (
    build_create_and_train_command,
    build_resume_command,
    build_train_current_command,
    write_run_config,
)

LOGGER = logging.getLogger(__name__)

_ALIAS_FALLBACKS = {
    "best": ("best", "final", "latest"),
    "latest": ("latest", "final", "best"),
    "final": ("final", "best", "latest"),
}

__all__ = ["DryRunStep", "ShowcaseExecutor", "format_dry_run_plan"]


@dataclass(frozen=True, slots=True)
class DryRunStep:
    """One planned showcase step for ``--dry-run`` output."""

    logical_name: str
    action: str
    parent: str | None
    env_id: str
    checkpoint_alias: str | None
    commands: tuple[str, ...]


class ShowcaseExecutor:
    """Run or dry-run a showcase plan."""

    def __init__(
        self,
        plan: ShowcasePlan,
        *,
        reuse: bool = False,
        from_node: str | None = None,
        force: bool = False,
    ) -> None:
        """Store plan options and load incremental run state when reusing."""
        self.plan = plan
        self.reuse = reuse
        self.from_node = from_node
        self.force = force
        self.exp_dir = plan.experiment_dir
        self.log_path = self.exp_dir / "showcase.log"
        self._run_state = self._load_or_init_state()
        self._video_cache: dict[str, list[str]] = dict(self._run_state.get("video_cache", {}))

    def run(self, *, dry_run: bool = False) -> int:
        """Execute the plan or print a dry-run transcript."""
        if dry_run:
            for line in format_dry_run_plan(self.plan):
                sys.stdout.write(f"{line}\n")
            return 0

        self._validate_workspace_policy()
        if self.plan.wandb_online and not _wandb_api_key_present():
            msg = "--wandb-online or large scale requires WANDB_API_KEY."
            raise RuntimeError(msg)

        self.exp_dir.mkdir(parents=True, exist_ok=True)
        resume_from = self.from_node is not None
        started_resume = False

        for node in self.plan.nodes:
            if self.from_node is not None and not started_resume:
                if node.logical_name != self.from_node:
                    continue
                started_resume = True

            node_state = self._node_state(node.logical_name)
            if self.reuse and node_state.get("status") == "completed":
                LOGGER.info("Skipping completed node %s", node.logical_name)
                continue
            if node_state.get("status") == "failed" and not resume_from and not self.force:
                msg = f"Node {node.logical_name!r} is failed. Use --from-node {node.logical_name} or --force to retry."
                raise RuntimeError(msg)

            self._mark_node(node.logical_name, status="running")
            started = time.perf_counter()
            try:
                if node.action == "root":
                    node_id = self._run_root(node)
                    checkpoint_step = None
                else:
                    node_id, checkpoint_step = self._run_child(node)
                duration = time.perf_counter() - started
                self._mark_node(
                    node.logical_name,
                    status="completed",
                    node_id=node_id,
                    checkpoint_step=checkpoint_step,
                    duration_seconds=duration,
                    exit_code=0,
                )
            except Exception as exc:
                duration = time.perf_counter() - started
                self._mark_node(
                    node.logical_name,
                    status="failed",
                    duration_seconds=duration,
                    exit_code=1,
                    error=str(exc),
                )
                LOGGER.exception("Showcase node %s failed", node.logical_name)
                raise

        write_showcase_summary(
            exp_dir=self.exp_dir,
            plan_meta={
                "experiment_name": self.plan.experiment_name,
                "scenario": self.plan.scenario.name,
                "scale": self.plan.scale.name,
                "algorithm": self.plan.algorithm,
                "workspace_root": str(self.plan.workspace_root),
                "wandb_online": self.plan.wandb_online,
            },
            run_state=self._run_state,
        )
        self._print_completion_message()
        return 0

    def _run_root(self, node: NodeSpec) -> str:
        node_state = self._node_state(node.logical_name)
        existing_id = node_state.get("node_id")
        if existing_id:
            return str(existing_id)

        config = build_node_run_config(self.plan, node)
        exit_code = self._subprocess_train(
            config=config,
            create_root=True,
            label=node.label,
        )
        if exit_code != 0:
            msg = f"Root training failed with exit code {exit_code}"
            raise RuntimeError(msg)
        graph = ExperimentGraph.from_directory(self.exp_dir, config_cls=RLRunConfig)
        return graph.current_node.id

    def _run_child(self, node: NodeSpec) -> tuple[str, int | None]:
        parent_state = self._node_state(node.parent or "")
        parent_node_id = parent_state.get("node_id")
        if not parent_node_id:
            msg = f"Parent state missing node_id for {node.parent!r}"
            raise ValueError(msg)

        graph = ExperimentGraph.from_directory(self.exp_dir, config_cls=RLRunConfig)
        parent_ws = graph.get_node(parent_node_id)
        checkpoint_ref = self._resolve_checkpoint(parent_ws, node.checkpoint_alias)
        if checkpoint_ref is None:
            msg = f"No checkpoint alias {node.checkpoint_alias!r} on parent {node.parent!r}"
            raise ValueError(msg)

        parent_env_id = graph.resolve_config(parent_ws).environment.env_id
        cache_key = f"{parent_node_id}:{checkpoint_ref.checkpoint_step}:{parent_env_id}"
        if cache_key not in self._video_cache:
            exit_code = render_checkpoint_video_subprocess(
                experiment_dir=self.exp_dir,
                node_id=parent_node_id,
                checkpoint_step=checkpoint_ref.checkpoint_step,
                env_id=parent_env_id,
                role="source_checkpoint",
                name_prefix=f"ckpt_{checkpoint_ref.checkpoint_step}_source",
                log_path=self.log_path,
                load_from_node_id=parent_node_id,
                parent_id=parent_node_id,
                child_id=None,
            )
            if exit_code != 0:
                msg = f"source_checkpoint video failed with exit code {exit_code}"
                raise RuntimeError(msg)
            self._video_cache[cache_key] = [cache_key]
            self._run_state["video_cache"] = self._video_cache
            self._append_node_video(node.parent or "", "source_checkpoint", cache_key)

        child_id = self._resolve_child_id(node, parent_node_id, graph, checkpoint_ref)
        child_config = build_node_run_config(self.plan, node)
        child_env_id = child_config.environment.env_id
        node_state = self._node_state(node.logical_name)
        child_videos = node_state.get("videos", {}).get("child_start", [])

        if not child_videos:
            exit_code = render_checkpoint_video_subprocess(
                experiment_dir=self.exp_dir,
                node_id=child_id,
                checkpoint_step=checkpoint_ref.checkpoint_step,
                env_id=child_env_id,
                role="child_start",
                name_prefix="child_start",
                log_path=self.log_path,
                use_parent_checkpoint=True,
                parent_id=parent_node_id,
                child_id=child_id,
            )
            if exit_code != 0:
                msg = f"child_start video failed with exit code {exit_code}"
                raise RuntimeError(msg)
            self._append_node_video(node.logical_name, "child_start", "child_start")

        graph = ExperimentGraph.from_directory(self.exp_dir, config_cls=RLRunConfig)
        child_ws = graph.get_node(child_id)
        resume = child_ws.resolve_resume_checkpoint_record() is not None
        self._mark_node(node.logical_name, node_id=child_id)
        exit_code = self._subprocess_train(
            config=child_config,
            create_root=False,
            node_id=child_id,
            resume=resume,
        )
        if exit_code != 0:
            tail = self._tail_log(self.log_path)
            msg = f"Training failed for {node.logical_name} with exit code {exit_code}"
            if tail:
                msg = f"{msg}\nLast log lines:\n{tail}"
            raise RuntimeError(msg)
        return child_id, checkpoint_ref.checkpoint_step

    def _resolve_child_id(
        self,
        node: NodeSpec,
        parent_node_id: str,
        graph: ExperimentGraph[RLRunConfig],
        checkpoint_ref: CheckpointRef,
    ) -> str:
        """Return an existing prepared/failed child or create a new one."""
        saved_id = self._node_state(node.logical_name).get("node_id")
        if isinstance(saved_id, str):
            try:
                workspace = graph.get_node(saved_id)
                if workspace.node_metadata.parent_id == parent_node_id:
                    return saved_id
            except KeyError:
                pass

        head = graph.branch_heads.get(node.branch)
        if (
            head is not None
            and head.id != parent_node_id
            and head.node_metadata.parent_id == parent_node_id
            and head.node_metadata.label == node.label
        ):
            return head.id

        child_id = self._create_child_node(node, parent_node_id, checkpoint_ref)
        self._mark_node(node.logical_name, node_id=child_id)
        return child_id

    def _create_child_node(self, node: NodeSpec, parent_node_id: str, checkpoint_ref: CheckpointRef) -> str:
        overrides = dict(node.overrides)
        overrides.setdefault("environment.env_id", node.env_id)
        if node.action == "extend":
            child_id = extend_branch(
                self.exp_dir,
                from_node=parent_node_id,
                branch=node.branch,
                label=node.label,
                config_overrides=overrides,
                from_checkpoint=checkpoint_ref,
                prepare=True,
            )
        else:
            child_id = fork_branch(
                self.exp_dir,
                from_node=parent_node_id,
                branch=node.branch,
                label=node.label,
                config_overrides=overrides,
                from_checkpoint=checkpoint_ref,
                prepare=True,
            )
        child_ws = ExperimentGraph.from_directory(self.exp_dir, config_cls=RLRunConfig).get_node(child_id)
        LOGGER.info(
            "Prepared %s (%s): parent=%s parent_checkpoint_step=%s env_id=%s",
            node.logical_name,
            child_id,
            child_ws.node_metadata.parent_id,
            child_ws.parent_checkpoint_step,
            overrides.get("environment.env_id", node.env_id),
        )
        return child_id

    def _subprocess_train(
        self,
        *,
        config: RLRunConfig,
        create_root: bool,
        label: str = "",
        node_id: str | None = None,
        resume: bool = False,
    ) -> int:
        config_path = self.exp_dir / f".showcase_{node_id or 'root'}_config.json"
        write_run_config(config, config_path)
        if create_root:
            command = build_create_and_train_command(
                experiment_dir=self.exp_dir,
                config_path=config_path,
                python_executable=sys.executable,
                label=label,
            )
        elif resume and node_id is not None:
            command = build_resume_command(
                experiment_dir=self.exp_dir,
                config_path=config_path,
                node_id=node_id,
                python_executable=sys.executable,
            )
        else:
            command = build_train_current_command(
                experiment_dir=self.exp_dir,
                config_path=config_path,
                python_executable=sys.executable,
                node_id=node_id,
            )

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n--- train {node_id or 'root'} ---\n")
            completed = subprocess.run(  # noqa: S603
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                env=_subprocess_env(),
            )
        return int(completed.returncode)

    @staticmethod
    def _tail_log(path: Path, *, max_lines: int = 20) -> str:
        if not path.is_file():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])

    def _resolve_checkpoint(self, workspace: NodeWorkspace, alias: str | None) -> CheckpointRef | None:
        if alias is None:
            return None
        for candidate in _ALIAS_FALLBACKS.get(alias, (alias,)):
            ref = checkpoint_ref_from_alias(workspace, candidate)
            if ref is not None:
                return ref
        return None

    def _validate_workspace_policy(self) -> None:
        manifest = self.exp_dir / MANIFEST_FILENAME
        if manifest.is_file() and not self.reuse and not self.force:
            msg = (
                f"Experiment directory already exists: {self.exp_dir}. "
                "Use --reuse to continue or --force to overwrite policy."
            )
            raise FileExistsError(msg)

    def _load_or_init_state(self) -> dict[str, Any]:
        if self.reuse and (self.exp_dir / SHOWCASE_RUN_FILENAME).is_file():
            return json.loads((self.exp_dir / SHOWCASE_RUN_FILENAME).read_text(encoding="utf-8"))
        return {
            "version": 1,
            "scenario": self.plan.scenario.name,
            "scale": self.plan.scale.name,
            "algorithm": self.plan.algorithm,
            "experiment_name": self.plan.experiment_name,
            "nodes": {},
            "video_cache": {},
        }

    def _node_state(self, logical_name: str) -> dict[str, Any]:
        nodes = self._run_state.setdefault("nodes", {})
        return nodes.setdefault(logical_name, {"status": "pending", "videos": {}})

    def _mark_node(self, logical_name: str, **fields: Any) -> None:
        state = self._node_state(logical_name)
        state.update(fields)
        state["updated_at"] = datetime.now(tz=UTC).isoformat()
        save_showcase_run(self.exp_dir, self._run_state)

    def _append_node_video(self, logical_name: str, role: str, tag: str) -> None:
        state = self._node_state(logical_name)
        videos = state.setdefault("videos", {})
        entries = videos.setdefault(role, [])
        if tag not in entries:
            entries.append(tag)

    def _print_completion_message(self) -> None:
        summary_path = self.exp_dir / "showcase_summary.json"
        from jarl.experiments.tensorboard import tensorboard_compare, tensorboard_lineage

        graph = ExperimentGraph.from_directory(self.exp_dir, config_cls=RLRunConfig)
        head = graph.current_node.id
        lines = [
            f"\nShowcase complete: {self.exp_dir}",
            f"Summary: {summary_path}",
            "Open in Runner Lab:",
            "  uv run streamlit run src/jarl/app/app.py",
            f"\nTensorBoard lineage:\n  tensorboard --logdir_spec {tensorboard_lineage(graph, head)}",
        ]
        node_ids = [state.get("node_id") for state in self._run_state.get("nodes", {}).values() if state.get("node_id")]
        if node_ids:
            lines.append(
                f"TensorBoard compare:\n  tensorboard --logdir_spec {tensorboard_compare(graph, node_ids[:5])}"
            )
        sys.stdout.write("\n".join(lines) + "\n")


def format_dry_run_plan(plan: ShowcasePlan) -> list[str]:
    """Return printable dry-run lines for the full plan."""
    lines = [
        f"Showcase dry-run: {plan.experiment_name} ({plan.scale.name}, {plan.algorithm})",
        f"Workspace: {plan.experiment_dir}",
        "",
    ]
    for node in plan.nodes:
        config = build_node_run_config(plan, node)
        lines.append(f"[{node.logical_name}] action={node.action} env={node.env_id} parent={node.parent}")
        if node.checkpoint_alias:
            lines.append(f"  checkpoint_alias: {node.checkpoint_alias}")
        if node.overrides:
            lines.append(f"  overrides: {json.dumps(node.overrides)}")
        lines.append(
            f"  timesteps={config.algorithm.total_timesteps} "
            f"nr_envs={config.environment.nr_envs} "
            f"nr_steps={config.algorithm.nr_steps} "
            f"minibatch={config.algorithm.minibatch_size} "
            f"algorithm={config.algorithm.name}"
        )
        if node.action == "root":
            lines.append("  command: jarl.training.run --create-root")
        else:
            lines.append("  command: extend/fork -> child_start video -> jarl.training.run train")
        lines.append("")
    return lines


def _wandb_api_key_present() -> bool:
    return bool(os.environ.get("WANDB_API_KEY"))


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORMS", "cuda,cpu")
    return env
