"""Shared constants for operation payloads."""

from __future__ import annotations

METRICS_SERIES_MAX_POINTS = 512
"""Default maximum points per metric series returned to agents."""

METRICS_SERIES_DEFAULT_KEYS: frozenset[str] = frozenset(
    {
        "rollout/episode_return",
        "rollout/episode_length",
        "eval/episode_return",
        "loss/policy_gradient_loss",
        "loss/critic_loss",
        "lr/learning_rate",
    }
)

SUBTREE_DEFAULT_DEPTH = 300
"""Default maximum depth for ``graph/subtree`` payloads."""

SUBTREE_MAX_VISIBLE = 2560
"""Default maximum nodes included in a subtree payload."""

CHECKPOINTS_DEFAULT_LIMIT = 20
"""Default maximum checkpoint candidates returned in search mode."""

CHECKPOINTS_MAX_LIMIT = 50
"""Hard cap on checkpoint candidates returned in one ``graph/checkpoints`` call."""

CHECKPOINT_ROLLOUT_MAX_STEPS = 4096
"""Hard cap on transitions captured by one checkpoint rollout operation."""

CHECKPOINT_ROLLOUT_RECORD_VIDEO_DEFAULT = False
"""Whether checkpoint rollout persists an MP4 when video fields are omitted."""

CHECKPOINT_ROLLOUT_VIDEO_VIEW_MODE_DEFAULT = "full"
"""Default RGB camera when checkpoint rollout records video."""

__all__ = [
    "CHECKPOINTS_DEFAULT_LIMIT",
    "CHECKPOINTS_MAX_LIMIT",
    "CHECKPOINT_ROLLOUT_MAX_STEPS",
    "CHECKPOINT_ROLLOUT_RECORD_VIDEO_DEFAULT",
    "CHECKPOINT_ROLLOUT_VIDEO_VIEW_MODE_DEFAULT",
    "METRICS_SERIES_DEFAULT_KEYS",
    "METRICS_SERIES_MAX_POINTS",
    "SUBTREE_DEFAULT_DEPTH",
    "SUBTREE_MAX_VISIBLE",
]
