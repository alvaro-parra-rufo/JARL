"""Experiment tree coordinator for branching training experiments."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar, overload

import networkx as nx

from jarl.config import BaseConfig, ConfigDiff
from jarl.envs.navix.catalog import assert_transfer_compatible, get_contract
from jarl.experiments.io.checkpoints import CheckpointRef
from jarl.experiments.io.layout import (
    CONFIG_FILENAME,
    NODE_METADATA_FILENAME,
    ExperimentLayout,
)
from jarl.experiments.manifest import (
    ExecutionState,
    ExperimentManifest,
)
from jarl.experiments.node import NodeMetadata, NodeStatus, NodeWorkspace
from jarl.experiments.reconcile import ReconcileReport
from jarl.experiments.run_config import TrackingConfig
from jarl.experiments.workspace_extension import resolve_workspace_class
from jarl.metadata import RunMetadata, collect_env_info, collect_git_info, now_iso, save_lockfile
from jarl.utils import dict_hash, short_uuid, write_text_atomic

__all__ = [
    "ExperimentGraph",
    "ReconcileReport",
]

NodeWorkspaceType = TypeVar("NodeWorkspaceType", bound=NodeWorkspace)


def _tracking_from_config(config: BaseConfig | None) -> TrackingConfig | None:
    """Return ``TrackingConfig`` embedded in a resolved run config, if any."""
    if config is None:
        return None
    tracking = getattr(config, "tracking", None)
    return tracking if isinstance(tracking, TrackingConfig) else None


class ExperimentGraph[ConfigT: BaseConfig]:
    """Central coordinator for tree-structured training experiments.

    Manages an experiment tree with branch heads, a persistent current node,
    lineage traversal, and config resolution. Each node is backed by a
    `NodeWorkspace` that handles per-node artifacts.

    Completed nodes are immutable; continuing training always creates a child
    via `extend` or `fork`. Failed or interrupted nodes may resume on the same
    node id. Use `prepare=True` to create nodes ready for deferred start.
    Use `checkout` to navigate without moving branch heads.

    Args:
        exp_dir: Root directory for the experiment.
        base_config: Immutable base configuration for the experiment root.
    """

    def __init__(self, exp_dir: str | Path, base_config: ConfigT | None = None) -> None:
        """Initialize the experiment graph."""
        self._dir = Path(exp_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._layout = ExperimentLayout(self._dir)
        self._tree: nx.DiGraph = nx.DiGraph()
        self._branch_heads: dict[str, str] = {}
        self._current_node_id: str | None = None
        self._execution_state = ExecutionState()
        self._nodes: dict[str, NodeWorkspace] = {}
        self._base_config: ConfigT | None = base_config
        self._metadata: RunMetadata | None = None

    def __repr__(self) -> str:
        """Return a string representation of the experiment graph."""
        n_nodes = self._tree.number_of_nodes()
        n_branches = len(self._branch_heads)
        return f"ExperimentGraph(dir={self._dir}, nodes={n_nodes}, branches={n_branches})"

    @property
    def layout(self) -> ExperimentLayout:
        """Path layout for this experiment root."""
        return self._layout

    def as_networkx(self) -> nx.DiGraph:
        """Return a copy of the experiment tree as a NetworkX digraph."""
        return self._tree.copy()

    # ---- Node lifecycle ---- #

    def create_root(
        self,
        config: ConfigT,
        branch: str = "main",
        label: str = "",
        description: str = "",
        metadata: dict[str, Any] | None = None,
        workspace_cls: type[NodeWorkspaceType] | None = None,
        lockfile: str | Path | None = None,
        *,
        prepare: bool = False,
    ) -> NodeWorkspace:
        """Create the root node of the experiment.

        Captures environment metadata, persists the base config, and initializes
        the tree with a single root node.

        Args:
            config: Base experiment config (becomes immutable root config).
            branch: Initial branch name.
            label: Optional human-readable label for the root node.
            description: Optional description of the experiment.
            metadata: Optional extensible metadata dict.
            workspace_cls: Custom `NodeWorkspace` subclass to use.
            lockfile: Optional path to a lockfile to copy for reproducibility.
            prepare: When ``True``, create the node in ``prepared`` status without
                opening training writers.

        Returns:
            The root `NodeWorkspace` (or subclass instance).

        Raises:
            RuntimeError: If a root node already exists.
        """
        if self._tree.number_of_nodes() > 0:
            raise RuntimeError("Experiment already has a root node.")

        files_to_check = [
            self._layout.config_path,
            self._layout.run_metadata_path,
            self._layout.manifest_path,
        ]
        if lockfile is not None:
            files_to_check.append(Path(lockfile))
        bootstrap_scaffold = frozenset({self._layout.config_path, self._layout.manifest_path})
        existing_files = [path for path in files_to_check if path.exists()]
        unexpected_files = [path.as_posix() for path in existing_files if path not in bootstrap_scaffold]
        if unexpected_files:
            raise RuntimeError(f"Experiment directory is not empty: {', '.join(unexpected_files)}")

        self._base_config = config
        config.save(self._layout.config_path)

        if self._metadata is None or not self._layout.run_metadata_path.exists():
            self._metadata = RunMetadata(
                start_time=now_iso(),
                env=collect_env_info(),
                git=collect_git_info(),
                transfer_group=_resolve_transfer_group(config),
            )
            self._metadata.save(self._layout.run_metadata_path)

        if lockfile is not None:
            save_lockfile(self._dir, lockfile)

        node_id = _build_node_id(branch, label)
        self._branch_heads[branch] = node_id
        ws = self._create_node(
            node_id=node_id,
            parent_id=None,
            branch=branch,
            label=label,
            description=description,
            config_overrides={},
            extra_metadata=metadata or {},
            workspace_cls=workspace_cls,
            prepared=prepare,
        )
        self._set_current_node(node_id)
        self.save()
        return ws

    def fork(
        self,
        branch: str,
        from_node: str | NodeWorkspace | None = None,
        *,
        config: ConfigT | None = None,
        config_diff: ConfigDiff | None = None,
        label: str = "",
        description: str = "",
        metadata: dict[str, Any] | None = None,
        workspace_cls: type[NodeWorkspaceType] | None = None,
        from_checkpoint: CheckpointRef | None = None,
        prepare: bool = False,
    ) -> NodeWorkspace:
        """Create a new branch from an existing node.

        Forks the training lineage at the given node, applying config overrides
        for the new branch. When ``from_node`` is omitted, the current node is
        used.

        Args:
            branch: Name for the new branch.
            from_node: Node to fork from. Defaults to the current node.
            config: Target config; sparse overrides are autocomputed as
                ``parent.diff(config).to_overrides()``.
            config_diff: Explicit delta precomputed as ``parent.diff(config)``.
            label: Optional human-readable label.
            description: Optional explanation of what/why.
            metadata: Optional extensible metadata dict.
            workspace_cls: Custom `NodeWorkspace` subclass.
            from_checkpoint: Optional parent checkpoint to restore from.
            prepare: When ``True``, create the node in ``prepared`` status without
                opening training writers.

        Returns:
            The new `NodeWorkspace` for the forked branch.

        Raises:
            KeyError: If the source node doesn't exist.
            ValueError: If the branch name already exists or both ``config`` and
                ``config_diff`` are provided.
            RuntimeError: If no current node is set when ``from_node`` is omitted.
        """
        parent_id = self._resolve_source_node_id(from_node)
        if parent_id not in self._tree:
            msg = f"Node not found: {parent_id}"
            raise KeyError(msg)
        if branch in self._branch_heads:
            msg = f"Branch already exists: {branch}"
            raise ValueError(msg)

        self._validate_config_transition(
            parent_id,
            config=config,
            config_diff=config_diff,
            from_checkpoint=from_checkpoint is not None,
        )

        overrides = self.resolve_config(parent_id).resolve_config_overrides(
            target=config,
            diff=config_diff,
        )
        parent_ws = self._nodes[parent_id]
        parent_ckpt_step = self._resolve_parent_checkpoint_step(
            parent_id=parent_id,
            parent_ws=parent_ws,
            from_checkpoint=from_checkpoint,
        )

        node_id = _build_node_id(branch, label)
        ws = self._create_node(
            node_id=node_id,
            parent_id=parent_id,
            branch=branch,
            label=label,
            description=description,
            config_overrides=overrides,
            extra_metadata=metadata or {},
            workspace_cls=workspace_cls,
            parent_ckpt_path=parent_ws.checkpoint_dir,
            parent_checkpoint_step=parent_ckpt_step,
            prepared=prepare,
        )
        self._branch_heads[branch] = node_id
        self._set_current_node(node_id)
        self.save()
        return ws

    def extend(
        self,
        branch: str | None = None,
        *,
        config: ConfigT | None = None,
        config_diff: ConfigDiff | None = None,
        label: str = "",
        description: str = "",
        metadata: dict[str, Any] | None = None,
        workspace_cls: type[NodeWorkspaceType] | None = None,
        from_checkpoint: CheckpointRef | None = None,
        prepare: bool = False,
    ) -> NodeWorkspace:
        """Add a new node continuing a branch from its head or the current node.

        When ``branch`` is omitted, the child is created from the current node
        on the current node's branch. When ``branch`` is provided, the child is
        appended to that branch's head.

        Args:
            branch: Branch to extend. Defaults to the current node's branch and
                uses the current node as parent.
            config: Target config; sparse overrides are autocomputed as
                ``parent.diff(config).to_overrides()``.
            config_diff: Explicit delta precomputed as ``parent.diff(config)``.
            label: Optional human-readable label.
            description: Optional explanation.
            metadata: Optional extensible metadata dict.
            workspace_cls: Custom `NodeWorkspace` subclass.
            from_checkpoint: Optional parent checkpoint to restore from.
            prepare: When ``True``, create the node in ``prepared`` status without
                opening training writers.

        Returns:
            The new head `NodeWorkspace`.

        Raises:
            KeyError: If the branch doesn't exist.
            ValueError: If both ``config`` and ``config_diff`` are provided.
            RuntimeError: If no current node is set when ``branch`` is omitted.
        """
        if branch is None:
            parent_id = self._require_current_node_id()
            branch = self._nodes[parent_id].branch
        else:
            if branch not in self._branch_heads:
                msg = f"Branch not found: {branch}"
                raise KeyError(msg)
            parent_id = self._branch_heads[branch]

        self._validate_config_transition(
            parent_id,
            config=config,
            config_diff=config_diff,
            from_checkpoint=from_checkpoint is not None,
        )

        parent_ws = self._nodes[parent_id]
        parent_ckpt_step = self._resolve_parent_checkpoint_step(
            parent_id=parent_id,
            parent_ws=parent_ws,
            from_checkpoint=from_checkpoint,
        )
        overrides = self.resolve_config(parent_id).resolve_config_overrides(
            target=config,
            diff=config_diff,
        )

        node_id = _build_node_id(branch, label)
        ws = self._create_node(
            node_id=node_id,
            parent_id=parent_id,
            branch=branch,
            label=label,
            description=description,
            config_overrides=overrides,
            extra_metadata=metadata or {},
            workspace_cls=workspace_cls,
            parent_ckpt_path=parent_ws.checkpoint_dir,
            parent_checkpoint_step=parent_ckpt_step,
            prepared=prepare,
        )
        self._branch_heads[branch] = node_id
        self._set_current_node(node_id)
        self.save()
        return ws

    def checkout(self, node: str | NodeWorkspace) -> None:
        """Move the current node pointer without changing branch heads.

        Args:
            node: Target node to check out.

        Raises:
            KeyError: If the node doesn't exist.
        """
        node_id = self._resolve_node_id(node)
        if node_id not in self._nodes:
            msg = f"Node not found: {node_id}"
            raise KeyError(msg)

        self._set_current_node(node_id)
        self.save()

    # ---- Lineage & config resolution ---- #

    @overload
    def resolve_config(self, node: str) -> ConfigT: ...
    @overload
    def resolve_config(self, node: NodeWorkspace) -> ConfigT: ...

    def resolve_config(self, node: str | NodeWorkspace) -> ConfigT:
        """Compute the fully resolved config for a node by walking its lineage.

        Starts from the base config and applies each ancestor's config overrides
        in order from root to the target node.

        Args:
            node: Target node (ID string or `NodeWorkspace` instance).

        Returns:
            Fully resolved config instance (same type as the graph's base config).

        Raises:
            RuntimeError: If no base config is set.
        """
        if self._base_config is None:
            msg = "No base config set."
            raise RuntimeError(msg)

        lineage = self.get_lineage(node)
        config = self._base_config
        for ws in lineage:
            overrides = ws.node_metadata.config_overrides
            if overrides:
                config = config.apply_overrides(overrides)
        return config

    @overload
    def get_lineage(self, node: str) -> list[NodeWorkspace]: ...
    @overload
    def get_lineage(self, node: NodeWorkspace) -> list[NodeWorkspace]: ...

    def get_lineage(self, node: str | NodeWorkspace) -> list[NodeWorkspace]:
        """Return the full path from root to the given node.

        Args:
            node: Target node (ID string or `NodeWorkspace` instance).

        Returns:
            Ordered list of `NodeWorkspace` instances from root to target (inclusive).

        Raises:
            KeyError: If the node doesn't exist.
        """
        node_id = self._resolve_node_id(node)
        if node_id not in self._tree:
            msg = f"Node not found: {node_id}"
            raise KeyError(msg)

        path = [node_id]
        current = node_id
        while True:
            parents = list(self._tree.predecessors(current))
            if not parents:
                break
            if len(parents) > 1:
                msg = f"Node {current} has multiple parents; tree structure required."
                raise RuntimeError(msg)
            current = parents[0]
            path.append(current)

        path.reverse()
        return [self._nodes[nid] for nid in path]

    @overload
    def get_metrics_along_lineage(self, node: str) -> list[dict[str, float]]: ...
    @overload
    def get_metrics_along_lineage(self, node: NodeWorkspace) -> list[dict[str, float]]: ...

    def get_metrics_along_lineage(self, node: str | NodeWorkspace) -> list[dict[str, float]]:
        """Collect latest metrics from each node along a lineage.

        Args:
            node: Target node (ID string or `NodeWorkspace` instance).

        Returns:
            List of metric dictionaries, one per node, root to target.
        """
        lineage = self.get_lineage(node)
        return [ws.latest_metrics() for ws in lineage]

    @overload
    def get_config_diff(self, node_a: str, node_b: str) -> ConfigDiff: ...
    @overload
    def get_config_diff(self, node_a: NodeWorkspace, node_b: NodeWorkspace) -> ConfigDiff: ...
    @overload
    def get_config_diff(self, node_a: str, node_b: NodeWorkspace) -> ConfigDiff: ...
    @overload
    def get_config_diff(self, node_a: NodeWorkspace, node_b: str) -> ConfigDiff: ...

    def get_config_diff(self, node_a: str | NodeWorkspace, node_b: str | NodeWorkspace) -> ConfigDiff:
        """Diff from ``node_a`` resolved config to ``node_b`` resolved config.

        Equivalent to ``resolve_config(node_a).diff(resolve_config(node_b))``.

        Args:
            node_a: Baseline node (parent side).
            node_b: Target node (child side).

        Returns:
            `ConfigDiff` from node_a config to node_b config.
        """
        config_a = self.resolve_config(node_a)
        config_b = self.resolve_config(node_b)
        return config_a.diff(config_b)

    # ---- Branch and current-node accessors ---- #

    @property
    def branch_heads(self) -> dict[str, NodeWorkspace]:
        """Return branch names mapped to their head workspaces."""
        return {name: self._nodes[nid] for name, nid in self._branch_heads.items()}

    def get_branches(self) -> dict[str, NodeWorkspace]:
        """Return branch names mapped to their head workspaces.

        Returns:
            Mapping from branch name to head `NodeWorkspace`.
        """
        return self.branch_heads

    def head(self, branch: str) -> NodeWorkspace:
        """Return the head node of a branch.

        Args:
            branch: Branch name.

        Returns:
            The head `NodeWorkspace`.

        Raises:
            KeyError: If the branch doesn't exist.
        """
        if branch not in self._branch_heads:
            msg = f"Branch not found: {branch}"
            raise KeyError(msg)
        return self._nodes[self._branch_heads[branch]]

    def is_branch_head(self, node: str | NodeWorkspace, *, branch: str | None = None) -> bool:
        """Return whether ``node`` is the head of ``branch``.

        When ``branch`` is omitted, the node's own branch is used.

        Args:
            node: Node id or workspace to check.
            branch: Branch name. Defaults to the node's branch.

        Returns:
            ``True`` when ``node`` is the head of the target branch.
        """
        workspace = self.get_node(self._resolve_node_id(node))
        target_branch = branch or workspace.branch
        return self.head(target_branch).id == workspace.id

    def require_branch_head(
        self,
        node: str | NodeWorkspace,
        *,
        branch: str | None = None,
    ) -> NodeWorkspace:
        """Return the node workspace when it is the head of ``branch``.

        Args:
            node: Node id or workspace that must be the branch head.
            branch: Branch name. Defaults to the node's branch.

        Returns:
            The validated `NodeWorkspace`.

        Raises:
            ValueError: If ``node`` is not the head of the target branch.
        """
        workspace = self.get_node(self._resolve_node_id(node))
        target_branch = branch or workspace.branch
        head = self.head(target_branch)
        if head.id != workspace.id:
            msg = f"Node {workspace.id!r} is not the head of branch {target_branch!r}; head is {head.id!r}."
            raise ValueError(msg)
        return workspace

    @property
    def current_node(self) -> NodeWorkspace:
        """Return the currently selected node workspace.

        Raises:
            RuntimeError: If no current node is set.
        """
        node_id = self._require_current_node_id()
        return self._nodes[node_id]

    @property
    def execution_state(self) -> ExecutionState:
        """Persisted execution snapshot for this experiment."""
        return self._execution_state

    def get_node(self, node_id: str) -> NodeWorkspace:
        """Retrieve a node workspace by its ID.

        Args:
            node_id: Node identifier.

        Returns:
            The `NodeWorkspace` instance.

        Raises:
            KeyError: If the node doesn't exist.
        """
        if node_id not in self._nodes:
            msg = f"Node not found: {node_id}"
            raise KeyError(msg)
        return self._nodes[node_id]

    def reconcile_stale_nodes(
        self,
        *,
        node_ids: Sequence[str] | None = None,
        assume_interrupted: bool = True,
        dry_run: bool = False,
    ) -> ReconcileReport:
        """Reconcile nodes left in ``training`` after a crashed or killed worker.

        Args:
            node_ids: Optional subset of node ids to inspect. Defaults to all nodes.
            assume_interrupted: When ``False``, only report candidates without mutating.
            dry_run: When ``True``, report candidates without writing metadata.

        Returns:
            Report listing examined nodes and those marked or eligible for interruption.
        """
        if not assume_interrupted and not dry_run:
            msg = "assume_interrupted=False requires dry_run=True."
            raise ValueError(msg)

        targets = tuple(node_ids) if node_ids is not None else tuple(self._nodes)
        examined: list[str] = []
        would_interrupt: list[str] = []
        interrupted: list[str] = []
        for node_id in targets:
            if node_id not in self._nodes:
                continue
            workspace = self._nodes[node_id]
            examined.append(node_id)
            if workspace.status != NodeStatus.TRAINING:
                continue
            would_interrupt.append(node_id)
            if dry_run or not assume_interrupted:
                continue
            workspace.mark_interrupted()
            interrupted.append(node_id)
        if interrupted:
            self.save()
        return ReconcileReport(
            examined=tuple(examined),
            would_interrupt=tuple(would_interrupt),
            interrupted=tuple(interrupted),
        )

    @property
    def base_config(self) -> ConfigT | None:
        """The immutable base config for this experiment."""
        return self._base_config

    @property
    def all_nodes(self) -> dict[str, NodeWorkspace]:
        """All nodes in the experiment."""
        return dict(self._nodes)

    # ---- Persistence ---- #

    def save(self) -> None:
        """Persist the experiment tree manifest atomically."""
        manifest = self._build_manifest()
        manifest.validate_tree()
        payload = manifest.model_dump(mode="json")
        write_text_atomic(
            self._layout.manifest_path,
            json.dumps(payload, indent=2, default=str),
        )

    @classmethod
    def from_directory(
        cls,
        exp_dir: str | Path,
        config_cls: type[ConfigT] = BaseConfig,
    ) -> ExperimentGraph[ConfigT]:
        """Reconstruct an experiment graph from an existing directory.

        Reads ``experiment.json``, ``config.json``, and all node directories to
        rebuild the full tree with workspaces.

        Args:
            exp_dir: Path to the experiment directory.
            config_cls: The `BaseConfig` subclass to use for deserialization.

        Returns:
            Reconstructed `ExperimentGraph`.

        Raises:
            FileNotFoundError: If the experiment directory or required files don't exist.
            ValueError: If the manifest is invalid or incomplete.
        """
        exp_dir = Path(exp_dir)
        if not exp_dir.exists():
            msg = f"Experiment directory not found: {exp_dir}"
            raise FileNotFoundError(msg)

        layout = ExperimentLayout(exp_dir)
        config_path = layout.config_path
        if not config_path.exists():
            msg = f"Base config not found: {config_path}"
            raise FileNotFoundError(msg)

        base_config = config_cls.load(config_path)
        graph = cls(exp_dir, base_config)

        manifest = _load_manifest(layout)
        manifest.validate_tree()
        graph._branch_heads = dict(manifest.branch_heads)
        graph._current_node_id = manifest.current_node or None
        graph._execution_state = manifest.execution_state
        graph._tree = nx.DiGraph()
        graph._tree.add_nodes_from(manifest.nodes)
        graph._tree.add_edges_from(manifest.edges)

        nodes_dir = layout.nodes_dir
        if nodes_dir.exists() and graph._tree.number_of_nodes() > 0:
            for node_id in nx.topological_sort(graph._tree):
                node_dir = layout.node_dir(node_id)
                meta_path = node_dir / NODE_METADATA_FILENAME
                if not meta_path.exists():
                    continue
                meta = NodeMetadata.load(meta_path)
                parent_ckpt = None
                if meta.parent_id and meta.parent_id in graph._nodes:
                    parent_ckpt = graph._nodes[meta.parent_id].checkpoint_dir
                ws_cls = resolve_workspace_class(meta.workspace_cls)
                node_config_path = node_dir / CONFIG_FILENAME
                node_tracking = None
                if node_config_path.exists():
                    node_tracking = _tracking_from_config(config_cls.load(node_config_path))
                ws = ws_cls(
                    node_dir=node_dir,
                    node_metadata=meta,
                    parent_checkpoint_path=parent_ckpt,
                    parent_checkpoint_step=meta.parent_checkpoint_step,
                    tracking=node_tracking,
                )
                graph._nodes[meta.id] = ws

        metadata_path = layout.run_metadata_path
        if metadata_path.exists():
            graph._metadata = RunMetadata.load(metadata_path)

        graph._validate_tree_invariants()
        return graph

    # ---- Internal ---- #

    def _create_node(
        self,
        node_id: str,
        parent_id: str | None,
        branch: str,
        label: str,
        description: str,
        config_overrides: dict[str, Any],
        extra_metadata: dict[str, Any],
        workspace_cls: type[NodeWorkspaceType] | None = None,
        parent_ckpt_path: Path | None = None,
        parent_checkpoint_step: int | None = None,
        *,
        prepared: bool = False,
    ) -> NodeWorkspace:
        """Create a node, add it to the tree, and return its workspace."""
        if node_id in self._nodes:
            raise RuntimeError(f"Node already exists: {node_id}")
        if parent_id is not None and parent_id not in self._nodes:
            raise RuntimeError(f"Parent node not found: {parent_id}")
        now = now_iso()
        ws_cls = workspace_cls if workspace_cls is not None else NodeWorkspace
        resolved_config = self._base_config
        if parent_id is not None:
            resolved_config = self.resolve_config(parent_id)
        if config_overrides and resolved_config is not None:
            resolved_config = resolved_config.apply_overrides(config_overrides)
        content_hash_input = {
            "resolved_config": resolved_config.model_dump() if resolved_config is not None else {},
            "parent_id": parent_id or "",
            "timestamp": now,
        }
        meta = NodeMetadata(
            id=node_id,
            parent_id=parent_id,
            branch=branch,
            label=label,
            description=description,
            metadata=extra_metadata,
            config_overrides=config_overrides,
            status=NodeStatus.PREPARED if prepared else NodeStatus.CREATED,
            created_at=now,
            updated_at=now,
            content_hash=dict_hash(content_hash_input),
            workspace_cls=f"{ws_cls.__module__}.{ws_cls.__qualname__}",
            parent_checkpoint_step=parent_checkpoint_step,
        )

        node_dir = self._layout.node_dir(node_id)

        ws = ws_cls(
            node_dir=node_dir,
            node_metadata=meta,
            parent_checkpoint_path=parent_ckpt_path,
            parent_checkpoint_step=parent_checkpoint_step,
            tracking=_tracking_from_config(resolved_config),
        )
        ws._save_metadata()

        if config_overrides:
            ws.save_config_overrides(config_overrides)

        if resolved_config is not None:
            ws.save_resolved_config(resolved_config)

        self._tree.add_node(node_id)
        if parent_id is not None:
            self._tree.add_edge(parent_id, node_id)

        self._nodes[node_id] = ws
        return ws

    def _resolve_parent_checkpoint_step(
        self,
        *,
        parent_id: str,
        parent_ws: NodeWorkspace,
        from_checkpoint: CheckpointRef | None,
    ) -> int | None:
        """Validate and pin a parent checkpoint reference when provided."""
        if from_checkpoint is None:
            return None
        if from_checkpoint.node_id != parent_id:
            msg = f"CheckpointRef node_id {from_checkpoint.node_id!r} does not match parent node {parent_id!r}."
            raise ValueError(msg)
        parent_ws.validate_saved_checkpoint(from_checkpoint.checkpoint_step)
        parent_ws.pin_checkpoint(from_checkpoint.checkpoint_step)
        return from_checkpoint.checkpoint_step

    def _build_manifest(self) -> ExperimentManifest:
        """Build the manifest snapshot for the current in-memory tree."""
        current_node = self._require_current_node_id()
        return ExperimentManifest(
            nodes=list(self._tree.nodes()),
            edges=list(self._tree.edges()),
            branch_heads=dict(self._branch_heads),
            current_node=current_node,
            execution_state=self._execution_state,
        )

    def _set_current_node(self, node_id: str) -> None:
        """Update the current node and execution snapshot."""
        self._current_node_id = node_id
        self._execution_state = ExecutionState(updated_at=now_iso(), status="idle")

    def _require_current_node_id(self) -> str:
        """Return the current node id or raise when unset."""
        if self._current_node_id is None:
            msg = "No current node is set for this experiment."
            raise RuntimeError(msg)
        return self._current_node_id

    def _resolve_source_node_id(self, from_node: str | NodeWorkspace | None) -> str:
        """Resolve an optional explicit source node to an id."""
        if from_node is None:
            return self._require_current_node_id()
        return self._resolve_node_id(from_node)

    def _validate_tree_invariants(self) -> None:
        """Validate runtime tree invariants."""
        for node_id in self._tree.nodes():
            parents = list(self._tree.predecessors(node_id))
            if len(parents) > 1:
                msg = f"Node {node_id} has multiple parents; tree structure required."
                raise RuntimeError(msg)

        for branch, head_id in self._branch_heads.items():
            if head_id not in self._nodes:
                msg = f"Branch head {branch!r} references unknown node {head_id!r}."
                raise RuntimeError(msg)

        if self._current_node_id is not None and self._current_node_id not in self._nodes:
            msg = f"Current node {self._current_node_id!r} is not present in loaded nodes."
            raise RuntimeError(msg)

    def _validate_config_transition(
        self,
        parent_id: str,
        *,
        config: ConfigT | None,
        config_diff: ConfigDiff | None,
        from_checkpoint: bool,
    ) -> None:
        """Reject immutable overrides and incompatible Navix map transfers."""
        del from_checkpoint
        if config is None and config_diff is None:
            return

        parent_config = self.resolve_config(parent_id)
        if config is not None:
            target_config = config
        else:
            overrides = config_diff.to_overrides() if config_diff is not None else {}
            target_config = parent_config.apply_overrides(overrides, nested=True)

        parent_config.validate_diff_allowed(parent_config.diff(target_config, flatten=True))
        self._validate_env_transfer(parent_config, target_config)

    def _validate_env_transfer(
        self,
        parent_config: ConfigT,
        target_config: ConfigT,
    ) -> None:
        """Validate Navix map compatibility when ``environment.env_id`` changes."""
        parent_env_id = _extract_env_id(parent_config)
        target_env_id = _extract_env_id(target_config)
        if parent_env_id is None or target_env_id is None or parent_env_id == target_env_id:
            return

        assert_transfer_compatible(parent_env_id, target_env_id)
        if self._metadata and self._metadata.transfer_group:
            target_group = get_contract(target_env_id).transfer_group
            if target_group != self._metadata.transfer_group:
                msg = (
                    f"Target env_id {target_env_id!r} belongs to transfer group "
                    f"{target_group!r}, but experiment is pinned to "
                    f"{self._metadata.transfer_group!r}."
                )
                raise ValueError(msg)

    @staticmethod
    def _resolve_node_id(node: str | NodeWorkspace) -> str:
        """Extract node ID from a string or NodeWorkspace."""
        if isinstance(node, NodeWorkspace):
            return node.id
        return node


def _load_manifest(layout: ExperimentLayout) -> ExperimentManifest:
    """Load a manifest from disk."""
    manifest_path = layout.manifest_path
    if not manifest_path.exists():
        msg = f"Experiment manifest not found: {manifest_path}"
        raise FileNotFoundError(msg)

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    edges = [tuple(edge) for edge in data.get("edges", [])]
    data["edges"] = edges
    return ExperimentManifest.model_validate(data)


def _build_node_id(branch: str, label: str = "", uuid_len: int = 8) -> str:
    """Construct a human-readable node ID.

    Format: ``{branch}_{label}_{uuid8}`` (label omitted if empty).
    """
    parts = [branch]
    if label:
        parts.append(label)
    parts.append(short_uuid(uuid_len))
    return "_".join(parts)


def _extract_env_id(config: BaseConfig) -> str | None:
    """Return ``environment.env_id`` when the config exposes an environment block."""
    environment = getattr(config, "environment", None)
    if environment is None:
        return None
    env_id = getattr(environment, "env_id", None)
    if env_id is None:
        return None
    return str(env_id)


def _resolve_transfer_group(config: BaseConfig) -> str:
    """Resolve the Navix transfer group pinned at experiment root creation."""
    env_id = _extract_env_id(config)
    if env_id is None:
        return ""
    return get_contract(env_id).transfer_group
