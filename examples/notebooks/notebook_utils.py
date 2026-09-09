"""Shared helpers for jarl experiment notebooks."""

from __future__ import annotations

import gzip
import html
import struct
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

if TYPE_CHECKING:
    from jarl.envs.navix.telemetry import NavixRolloutSummary
    from jarl.experiments.graph import ExperimentGraph
    from jarl.experiments.node import NodeWorkspace
    from jarl.experiments.run_config import CheckpointConfig, RunConfig

_IDX_IMAGE_MAGIC = 2051
_IDX_LABEL_MAGIC = 2049
_MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

__all__ = [
    "MNISTMLP",
    "MNISTBatch",
    "MNISTPlayground",
    "display_rollout_analysis",
    "display_tensorboard",
    "evaluate_mnist",
    "init_mnist_params",
    "load_mnist_arrays",
    "load_warmstart_params",
    "print_branch_summary",
    "resolve_fork_step",
    "sample_minibatch",
    "train_from_config",
    "train_mnist_node",
]


class MNISTMLP(nn.Module):
    """Small fully connected classifier for MNIST demos."""

    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return class logits for a batch of grayscale images."""
        flat = x.reshape((x.shape[0], -1)) / 255.0
        hidden = nn.relu(nn.Dense(self.hidden_dim)(flat))
        return nn.Dense(10)(hidden)


@dataclass(frozen=True, slots=True)
class MNISTBatch:
    """MNIST images and labels stored as NumPy arrays."""

    images: np.ndarray
    labels: np.ndarray


def _download_mnist_file(name: str, data_root: Path) -> Path:
    data_root.mkdir(parents=True, exist_ok=True)
    filename = _MNIST_FILES[name]
    destination = data_root / filename
    if not destination.exists():
        urllib.request.urlretrieve(f"{_MNIST_BASE_URL}{filename}", destination)  # noqa: S310
    return destination


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != _IDX_IMAGE_MAGIC:
            msg = f"Unexpected MNIST image magic number in {path}"
            raise ValueError(msg)
        buffer = handle.read(count * rows * cols)
    return np.frombuffer(buffer, dtype=np.uint8).reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != _IDX_LABEL_MAGIC:
            msg = f"Unexpected MNIST label magic number in {path}"
            raise ValueError(msg)
        buffer = handle.read(count)
    return np.frombuffer(buffer, dtype=np.uint8).astype(np.int32)


def load_mnist_arrays(
    *, train_size: int = 2048, test_size: int = 512, data_root: str = "./_data"
) -> tuple[MNISTBatch, MNISTBatch]:
    """Load MNIST subsets as NumPy arrays for fast notebook demos.

    Args:
        train_size: Number of training examples to keep.
        test_size: Number of test examples to keep.
        data_root: Download directory for MNIST IDX gzip files.

    Returns:
        Training and test batches as ``MNISTBatch`` instances.
    """
    root = Path(data_root)
    train_images = _read_idx_images(_download_mnist_file("train_images", root))[:train_size]
    train_labels = _read_idx_labels(_download_mnist_file("train_labels", root))[:train_size]
    test_images = _read_idx_images(_download_mnist_file("test_images", root))[:test_size]
    test_labels = _read_idx_labels(_download_mnist_file("test_labels", root))[:test_size]
    return (
        MNISTBatch(images=train_images, labels=train_labels),
        MNISTBatch(images=test_images, labels=test_labels),
    )


def init_mnist_params(rng: jax.Array, *, hidden_dim: int) -> dict[str, Any]:
    """Initialize Flax parameters for ``MNISTMLP``."""
    model = MNISTMLP(hidden_dim=hidden_dim)
    dummy = jnp.zeros((1, 28, 28), dtype=jnp.float32)
    return model.init(rng, dummy)


def sample_minibatch(batch: MNISTBatch, *, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample a random minibatch from ``batch``."""
    indices = rng.choice(batch.images.shape[0], size=batch_size, replace=False)
    return batch.images[indices], batch.labels[indices]


def evaluate_mnist(params: dict[str, Any], test_batch: MNISTBatch, *, hidden_dim: int) -> float:
    """Return classification accuracy on ``test_batch``."""
    model = MNISTMLP(hidden_dim=hidden_dim)
    logits = model.apply(params, jnp.asarray(test_batch.images))
    predictions = jnp.argmax(logits, axis=-1)
    return float(jnp.mean(predictions == jnp.asarray(test_batch.labels)))


def load_warmstart_params(workspace: NodeWorkspace, *, from_parent: bool = False) -> dict[str, Any]:
    """Load checkpoint params for fork/extend warm-starts."""
    state = workspace.load_parent_checkpoint() if from_parent else workspace.load_checkpoint()
    if not isinstance(state, dict) or "params" not in state:
        msg = "Expected checkpoint state dict with a 'params' entry."
        raise TypeError(msg)
    return jax.device_get(state["params"])


def resolve_fork_step(workspace: NodeWorkspace, *, preferred: int = 4) -> int:
    """Pick a restorable checkpoint step for forks, preferring ``preferred``."""
    registered = sorted({record.checkpoint_step for record in workspace.list_checkpoints()})
    preferred_first = [step for step in registered if step >= preferred]
    fallback = list(reversed([step for step in registered if step < preferred]))
    errors: list[str] = []

    for step in preferred_first + fallback:
        try:
            workspace.validate_saved_checkpoint(step)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        return step

    msg = "No restorable checkpoint found for fork."
    if errors:
        msg = f"{msg} Attempts: {' | '.join(errors)}"
    raise RuntimeError(msg)


def train_from_config(
    workspace: NodeWorkspace,
    config: RunConfig,
    train_batch: MNISTBatch,
    test_batch: MNISTBatch,
    *,
    train_steps: int | None = None,
    params: dict[str, Any] | None = None,
    start_step: int = 0,
    fail_at_step: int | None = None,
    register_model: bool = True,
) -> dict[str, Any]:
    """Train using hyperparameters stored on a MNIST ``RunConfig`` subclass."""
    fields = config.model_dump()
    steps = train_steps if train_steps is not None else int(fields["train_steps"])
    return train_mnist_node(
        workspace,
        config=config,
        learning_rate=float(fields["learning_rate"]),
        train_steps=steps,
        hidden_dim=int(fields["hidden_dim"]),
        batch_size=int(fields["batch_size"]),
        train_batch=train_batch,
        test_batch=test_batch,
        params=params,
        start_step=start_step,
        fail_at_step=fail_at_step,
        register_model=register_model,
    )


def print_branch_summary(graph: object, *, metric_key: str = "test_accuracy") -> None:
    """Print branch heads with resolved config and latest metrics."""
    from jarl.experiments.graph import ExperimentGraph

    if not isinstance(graph, ExperimentGraph):
        msg = "graph must be an ExperimentGraph instance."
        raise TypeError(msg)

    rows: list[tuple[str, str, str, str, str]] = []
    for branch_name, head in sorted(graph.get_branches().items()):
        cfg = graph.resolve_config(head)
        fields = cfg.model_dump()
        metrics = head.latest_metrics()
        accuracy = metrics.get(metric_key)
        acc_text = f"{accuracy:.3f}" if accuracy is not None else "n/a"
        rows.append(
            (
                branch_name,
                head.id,
                f"lr={fields['learning_rate']}",
                f"h={fields['hidden_dim']}",
                acc_text,
            )
        )

    width = max(len(row[1]) for row in rows) if rows else 20
    print(f"{'branch':<12} {'node_id':<{width}} {'lr':<10} {'hidden':<8} test_acc")
    print("-" * (12 + width + 10 + 8 + 10))
    for branch_name, node_id, lr_text, hidden_text, acc_text in rows:
        print(f"{branch_name:<12} {node_id:<{width}} {lr_text:<10} {hidden_text:<8} {acc_text}")


def display_tensorboard(logdir_spec: str, *, port: int = 6006) -> None:
    """Show TensorBoard inline in Jupyter or open it in a browser iframe."""
    if not logdir_spec.strip():
        print("TensorBoard: empty logdir_spec — train at least one branch first.")
        return

    missing: list[str] = []
    for entry in logdir_spec.split(","):
        if ":" not in entry:
            continue
        _label, path_str = entry.split(":", 1)
        tb_dir = Path(path_str)
        has_events = tb_dir.is_dir() and any(tb_dir.glob("events.out.tfevents.*"))
        if not has_events:
            missing.append(entry)

    if missing:
        print("TensorBoard: no event files yet for:")
        for entry in missing:
            print(f"  - {entry}")
        print("Re-train with track_tensorboard=True (playground default) after updating jarl.")

    try:
        ipython = get_ipython()  # type: ignore[name-defined]
    except NameError:
        ipython = None

    if ipython is not None and hasattr(ipython, "run_line_magic"):
        ipython.run_line_magic("load_ext", "tensorboard")
        ipython.run_line_magic("tensorboard", f"--logdir_spec={logdir_spec} --port={port}")
        return

    from IPython.display import IFrame, display

    from jarl.experiments.tensorboard import launch_tensorboard

    launch_tensorboard(logdir_spec, port=port)
    display(IFrame(src=f"http://localhost:{port}", width="100%", height=600))


def display_rollout_analysis(
    summary: NavixRolloutSummary,
    video_path: str | Path | None,
    *,
    cache_hit: bool | None = None,
) -> None:
    """Display a rollout video and compact metrics in two columns.

    Args:
        summary: Deterministic Navix rollout summary.
        video_path: Optional MP4 artifact path.
        cache_hit: Optional cache status shown with rollout identity.
    """
    from IPython.display import HTML, Video, display

    path = Path(video_path) if video_path is not None else None
    if path is not None and path.is_file():
        video_markup = Video(
            filename=str(path),
            embed=True,
            html_attributes="controls loop",
        )._repr_html_()
    else:
        video_markup = "<div class='jarl-empty'>No hay vídeo materializado para este rollout.</div>"

    outcome = summary.outcome
    identity = summary.identity
    success = "—" if outcome.success is None else ("Sí" if outcome.success else "No")
    cache_label = "—" if cache_hit is None else ("hit" if cache_hit else "miss")
    action_rows = "".join(
        f"<tr><td>{html.escape(item.name)}</td><td>{item.count}</td></tr>" for item in summary.actions.counts
    )
    event_rows = "".join(
        f"<tr><td>{html.escape(item.kind)}</td><td>{html.escape(item.source)}</td><td>{item.count}</td></tr>"
        for item in summary.events.counts
    )
    visibility_rows = "".join(
        f"<tr><td>{html.escape(item.entity_name)}</td><td>{html.escape(item.formatted_intervals or '—')}</td></tr>"
        for item in summary.visibility
        if item.ever_seen
    )
    panel = f"""
    <style>
      .jarl-rollout-grid {{
        color-scheme: light dark;
        --jarl-bg: var(--jp-layout-color1, var(--vscode-editor-background, Canvas));
        --jarl-surface: var(--jp-layout-color2, var(--vscode-editorWidget-background, Canvas));
        --jarl-text: var(--jp-ui-font-color1, var(--vscode-editor-foreground, CanvasText));
        --jarl-muted: var(--jp-ui-font-color2, var(--vscode-descriptionForeground, GrayText));
        --jarl-border: var(--jp-border-color2, var(--vscode-panel-border, ButtonBorder));
        display: grid;
        grid-template-columns: minmax(320px, 1.15fr) minmax(300px, 0.85fr);
        gap: 18px;
        align-items: start;
        color: var(--jarl-text);
      }}
      .jarl-rollout-card {{
        border: 1px solid var(--jarl-border);
        border-radius: 10px;
        padding: 14px;
        background: var(--jarl-bg);
        color: var(--jarl-text);
      }}
      .jarl-rollout-card video {{ width: 100%; max-height: 520px; }}
      .jarl-rollout-card h4 {{ color: var(--jarl-text); }}
      .jarl-metrics {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
        margin-bottom: 12px;
      }}
      .jarl-metric {{
        border-radius: 8px;
        padding: 9px;
        background: var(--jarl-surface);
        color: var(--jarl-text);
      }}
      .jarl-metric small {{ color: var(--jarl-muted); }}
      .jarl-rollout-card table {{
        width: 100%;
        border-collapse: collapse;
        margin: 7px 0 13px;
        color: var(--jarl-text);
      }}
      .jarl-rollout-card th, .jarl-rollout-card td {{
        text-align: left;
        padding: 4px 6px;
        border-bottom: 1px solid var(--jarl-border);
      }}
      .jarl-identity {{ color: var(--jarl-muted); font-size: 0.88em; line-height: 1.5; }}
      .jarl-empty {{ padding: 30px; text-align: center; color: var(--jarl-muted); }}
      @media (max-width: 900px) {{
        .jarl-rollout-grid {{ grid-template-columns: 1fr; }}
      }}
    </style>
    <div class="jarl-rollout-grid">
      <div class="jarl-rollout-card">{video_markup}</div>
      <div class="jarl-rollout-card">
        <div class="jarl-metrics">
          <div class="jarl-metric"><small>Return</small><br><b>{outcome.return_total:.3f}</b></div>
          <div class="jarl-metric"><small>Pasos</small><br><b>{outcome.length}</b></div>
          <div class="jarl-metric"><small>Éxito</small><br><b>{success}</b></div>
          <div class="jarl-metric"><small>Final</small><br><b>{html.escape(outcome.reason)}</b></div>
        </div>
        <div class="jarl-identity">
          <b>{html.escape(identity.env_id)}</b><br>
          node={html.escape(identity.node_id or "—")} · checkpoint={identity.checkpoint_step} ·
          seed={identity.seed} · cache={cache_label}<br>
          rollout={html.escape(identity.rollout_id or "—")}
        </div>
        <h4>Acciones</h4>
        <table><tr><th>Acción</th><th>Total</th></tr>{action_rows}</table>
        <h4>Eventos</h4>
        <table><tr><th>Evento</th><th>Fuente</th><th>Total</th></tr>{event_rows}</table>
        <h4>Visibilidad</h4>
        <table><tr><th>Entidad</th><th>Intervalos</th></tr>{visibility_rows}</table>
      </div>
    </div>
    """
    display(HTML(panel))


class MNISTPlayground:
    """Small interactive API for MNIST experiment trees in notebooks."""

    def __init__(
        self,
        exp_dir: str | Path,
        *,
        train_batch: MNISTBatch,
        test_batch: MNISTBatch,
        config_cls: type[RunConfig],
        base_config: RunConfig | None = None,
    ) -> None:
        """Configure paths, data batches, and default run settings for the playground."""
        self.exp_dir = Path(exp_dir)
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.config_cls = config_cls
        self._base_config = base_config or config_cls()
        self.graph: ExperimentGraph | None = None

    def default_config(self, **overrides: object) -> RunConfig:
        """Build a config from the playground defaults plus optional overrides."""
        return self._base_config.model_copy(update=overrides)

    def reset(self) -> None:
        """Delete the experiment directory and clear the in-memory graph."""
        import shutil

        if self.exp_dir.exists():
            shutil.rmtree(self.exp_dir)
        self.graph = None

    def load(self) -> ExperimentGraph:
        """Load an existing experiment from disk or create a baseline root."""
        from jarl.experiments.graph import ExperimentGraph

        manifest = self.exp_dir / "experiment.json"
        if manifest.exists():
            self.graph = ExperimentGraph.from_directory(self.exp_dir, config_cls=self.config_cls)
            return self.graph

        self.exp_dir.mkdir(parents=True, exist_ok=True)
        config = self.default_config()
        self.graph = ExperimentGraph(self.exp_dir, base_config=config)
        self.graph.create_root(
            config=config,
            branch="main",
            label="baseline",
            description="Playground baseline",
        )
        return self.graph

    @property
    def graph_or_load(self) -> ExperimentGraph:
        """Return the active graph, loading from disk when needed."""
        if self.graph is None:
            return self.load()
        return self.graph

    def branch_names(self) -> list[str]:
        """Return sorted branch names for UI selectors."""
        return sorted(self.graph_or_load.get_branches())

    def train_head(
        self,
        branch: str,
        *,
        learning_rate: float | None = None,
        hidden_dim: int | None = None,
        batch_size: int | None = None,
        train_steps: int | None = None,
    ) -> NodeWorkspace:
        """Train the current head of ``branch`` with optional hyperparameter overrides."""
        graph = self.graph_or_load
        head = graph.head(branch)
        if head.status.value == "completed":
            msg = f"Branch '{branch}' head is completed. Call extend_branch() to continue."
            raise RuntimeError(msg)

        cfg = graph.resolve_config(head)
        updates: dict[str, object] = {}
        if learning_rate is not None:
            updates["learning_rate"] = learning_rate
        if hidden_dim is not None:
            updates["hidden_dim"] = hidden_dim
        if batch_size is not None:
            updates["batch_size"] = batch_size
        if train_steps is not None:
            updates["train_steps"] = train_steps
        run_cfg = cfg.model_copy(update=updates) if updates else cfg

        warmstart = None
        if head.status.value in {"failed", "interrupted"}:
            with head:
                state = head.load_resume_checkpoint()
                warmstart = state["params"] if isinstance(state, dict) else state
        elif head.node_metadata.parent_id is not None and head.status.value == "created":
            warmstart = load_warmstart_params(head, from_parent=True)
        elif head.status.value == "created" and head.node_metadata.parent_id is None:
            warmstart = None

        with head:
            train_from_config(
                head,
                run_cfg,
                self.train_batch,
                self.test_batch,
                params=warmstart,
            )
        graph.save()
        return head

    def fork_branch(
        self,
        from_branch: str,
        new_branch: str,
        *,
        label: str = "",
        learning_rate: float | None = None,
        hidden_dim: int | None = None,
        batch_size: int | None = None,
        train_steps: int | None = None,
        train: bool = True,
    ) -> NodeWorkspace:
        """Fork ``from_branch`` into ``new_branch`` and optionally train it."""
        graph = self.graph_or_load
        parent = graph.head(from_branch)
        fork_step = resolve_fork_step(parent, preferred=4)
        parent.pin_checkpoint(fork_step)

        parent_cfg = graph.resolve_config(parent)
        updates: dict[str, object] = {}
        if learning_rate is not None:
            updates["learning_rate"] = learning_rate
        if hidden_dim is not None:
            updates["hidden_dim"] = hidden_dim
        if batch_size is not None:
            updates["batch_size"] = batch_size
        if train_steps is not None:
            updates["train_steps"] = train_steps
        child_cfg = parent_cfg.model_copy(update=updates) if updates else parent_cfg

        from jarl.experiments import CheckpointRef

        parent_hidden = int(parent_cfg.model_dump()["hidden_dim"])
        child_hidden = int(child_cfg.model_dump()["hidden_dim"])
        warmstart = child_hidden == parent_hidden
        node = graph.fork(
            new_branch,
            from_node=parent,
            config=child_cfg,
            label=label or new_branch,
            description=f"Playground fork from {from_branch}@{fork_step}",
            from_checkpoint=CheckpointRef(node_id=parent.id, checkpoint_step=fork_step),
        )
        if train:
            with node:
                train_from_config(
                    node,
                    child_cfg,
                    self.train_batch,
                    self.test_batch,
                    params=load_warmstart_params(node) if warmstart else None,
                )
        graph.save()
        return node

    def extend_branch(
        self,
        branch: str,
        *,
        label: str = "continue",
        learning_rate: float | None = None,
        hidden_dim: int | None = None,
        batch_size: int | None = None,
        train_steps: int | None = None,
        train: bool = True,
    ) -> NodeWorkspace:
        """Append a child node on ``branch`` and optionally train it."""
        graph = self.graph_or_load
        head_cfg = graph.resolve_config(graph.head(branch))
        updates: dict[str, object] = {}
        if learning_rate is not None:
            updates["learning_rate"] = learning_rate
        if hidden_dim is not None:
            updates["hidden_dim"] = hidden_dim
        if batch_size is not None:
            updates["batch_size"] = batch_size
        if train_steps is not None:
            updates["train_steps"] = train_steps
        child_cfg = head_cfg.model_copy(update=updates) if updates else head_cfg

        node = graph.extend(branch, config=child_cfg, label=label, description=f"Playground extend on {branch}")
        if train:
            with node:
                train_from_config(
                    node,
                    graph.resolve_config(node),
                    self.train_batch,
                    self.test_batch,
                    params=load_warmstart_params(node, from_parent=True),
                )
        graph.save()
        return node

    def checkout(self, branch: str) -> NodeWorkspace:
        """Move the current-node pointer to the head of ``branch``."""
        graph = self.graph_or_load
        head = graph.head(branch)
        graph.checkout(head)
        graph.save()
        return head

    def tensorboard_spec(self, branches: list[str] | None = None) -> str:
        """Build a TensorBoard ``--logdir_spec`` for selected branches."""
        from jarl.experiments.tensorboard import tensorboard_compare, tensorboard_lineage

        graph = self.graph_or_load
        if branches is None:
            branches = self.branch_names()
        if len(branches) == 1:
            return tensorboard_lineage(graph, graph.head(branches[0]))
        return tensorboard_compare(graph, branches)

    def show_tensorboard(self, branches: list[str] | None = None, *, port: int = 6006) -> None:
        """Open TensorBoard in the notebook for the selected branches."""
        display_tensorboard(self.tensorboard_spec(branches), port=port)

    def plot_tree(self, *, metric_key: str = "test_accuracy") -> object:
        """Plot the experiment DAG colored by ``metric_key``."""
        import matplotlib.pyplot as plt

        from jarl.experiments import plot_dag

        fig = plot_dag(self.graph_or_load, metric_key=metric_key, figsize=(12, 8))
        plt.show()
        return fig

    def status(self) -> None:
        """Print branch heads, configs, and latest metrics."""
        graph = self.graph_or_load
        print(f"Experiment: {self.exp_dir.resolve()}")
        print(f"Current node: {graph.current_node.id} ({graph.current_node.branch})")
        print_branch_summary(graph)


def train_mnist_node(
    workspace: NodeWorkspace,
    *,
    config: RunConfig,
    learning_rate: float,
    train_steps: int,
    hidden_dim: int,
    batch_size: int,
    train_batch: MNISTBatch,
    test_batch: MNISTBatch,
    params: dict[str, Any] | None = None,
    start_step: int = 0,
    checkpoint_policy: CheckpointConfig | None = None,
    fail_at_step: int | None = None,
    register_model: bool = True,
) -> dict[str, Any]:
    """Run a short MNIST training loop through ``NodeWorkspace`` IO APIs.

    Args:
        workspace: Active node workspace inside a training context manager.
        config: Run configuration providing checkpoint and tracking settings.
        learning_rate: Adam learning rate for this node.
        train_steps: Number of optimizer steps to run.
        hidden_dim: Hidden layer width for ``MNISTMLP``.
        batch_size: Minibatch size.
        train_batch: Training subset.
        test_batch: Evaluation subset.
        params: Optional warm-start parameters.
        start_step: Step offset when resuming training.
        checkpoint_policy: Checkpoint policy override for ``save_checkpoint_if_due``.
        fail_at_step: Raise ``RuntimeError`` when this step is reached.
        register_model: Persist a ``policy`` model archive when training succeeds.

    Returns:
        Final Flax parameter tree.
    """
    from jarl.experiments.io.model_archive import build_model_artifact_record, save_model_archive

    policy = checkpoint_policy or config.checkpoint
    rng = np.random.default_rng(config.model_dump().get("seed", 0))
    if params is None:
        params = init_mnist_params(jax.random.key(int(rng.integers(0, 1_000_000))), hidden_dim=hidden_dim)

    model = MNISTMLP(hidden_dim=hidden_dim)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    workspace.init_checkpoint_manager(max_to_keep=policy.max_to_keep, save_interval_steps=policy.save_interval_steps)
    started = time.monotonic()

    @jax.jit
    def train_step(
        step_params: dict[str, Any],
        step_opt_state: optax.OptState,
        x_batch: jax.Array,
        y_batch: jax.Array,
    ) -> tuple[dict[str, Any], optax.OptState, jax.Array]:
        def loss_fn(current_params: dict[str, Any]) -> jax.Array:
            logits = model.apply(current_params, x_batch)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y_batch).mean()

        loss, grads = jax.value_and_grad(loss_fn)(step_params)
        updates, next_opt_state = optimizer.update(grads, step_opt_state)
        next_params = optax.apply_updates(step_params, updates)
        return next_params, next_opt_state, loss

    for step in range(start_step + 1, train_steps + 1):
        x_np, y_np = sample_minibatch(train_batch, batch_size=batch_size, rng=rng)
        params, opt_state, loss = train_step(
            params,
            opt_state,
            jnp.asarray(x_np),
            jnp.asarray(y_np),
        )
        accuracy = evaluate_mnist(params, test_batch, hidden_dim=hidden_dim)
        workspace.log_scalars(step, train_loss=float(loss), test_accuracy=accuracy)
        elapsed = time.monotonic() - started
        state = {"params": params, "step": step}
        workspace.save_checkpoint_if_due(
            step,
            state,
            policy=policy,
            total_steps=train_steps,
            elapsed_seconds=elapsed,
            metrics={"test_accuracy": accuracy},
        )
        if fail_at_step is not None and step == fail_at_step:
            msg = f"Simulated training failure at step {step}"
            raise RuntimeError(msg)

    if register_model:
        workspace.models_dir.mkdir(parents=True, exist_ok=True)
        archive_path = save_model_archive(
            workspace.models_dir,
            name="policy",
            step=train_steps,
            components={"params": jax.device_get(params)},
        )
        record = build_model_artifact_record(name="policy", step=train_steps, filename=archive_path.name)
        workspace.register_artifact(record)
    return params
