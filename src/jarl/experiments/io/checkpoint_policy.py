"""Checkpoint save and Orbax preservation policy helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import orbax.checkpoint as ocp

from jarl.experiments.run_config import CheckpointConfig

__all__ = [
    "CheckpointPolicyState",
    "PinnedStepsPreservationPolicy",
    "build_checkpoint_preservation_policy",
    "should_save_checkpoint",
]


@dataclass
class CheckpointPolicyState:
    """Mutable state tracked across training for checkpoint policies.

    Args:
        last_save_monotonic: Monotonic clock timestamp of the last save.
        saved_progress_fractions: Progress milestones already checkpointed.
    """

    last_save_monotonic: float = 0.0
    saved_progress_fractions: set[float] = field(default_factory=set)


def should_save_checkpoint(
    step: int,
    *,
    config: CheckpointConfig,
    state: CheckpointPolicyState,
    total_steps: int | None,
    elapsed_seconds: float | None,
    monotonic_now: float,
) -> bool:
    """Return whether a checkpoint should be saved for the current step.

    Args:
        step: Current node-relative training step.
        config: Checkpoint configuration with interval policies.
        state: Mutable policy state updated when this function returns ``True``.
        total_steps: Planned total steps for percentage milestones.
        elapsed_seconds: Elapsed training time for time-based policies.
        monotonic_now: Current monotonic clock reading in seconds.

    Returns:
        ``True`` when any enabled policy requests a checkpoint at this step.
    """
    if step <= 0:
        return False

    if config.save_interval_steps > 0 and step % config.save_interval_steps == 0:
        state.last_save_monotonic = monotonic_now
        return True

    if (
        config.save_interval_seconds is not None
        and elapsed_seconds is not None
        and (
            state.last_save_monotonic == 0.0
            or (monotonic_now - state.last_save_monotonic) >= config.save_interval_seconds
        )
    ):
        state.last_save_monotonic = monotonic_now
        return True

    if config.save_at_progress_fractions and total_steps is not None and total_steps > 0:
        progress = step / total_steps
        for fraction in sorted(config.save_at_progress_fractions):
            if fraction in state.saved_progress_fractions:
                continue
            if progress >= fraction:
                state.saved_progress_fractions.add(fraction)
                state.last_save_monotonic = monotonic_now
                return True

    return False


@dataclass
class PinnedStepsPreservationPolicy:
    """Preserve Orbax steps referenced by downstream node forks.

    Args:
        get_pinned_steps: Callable returning pinned Orbax step identifiers.
    """

    get_pinned_steps: Callable[[], set[int]]

    def should_preserve(
        self,
        checkpoints: Sequence[ocp.checkpoint_managers.PolicyCheckpointInfo],
        *,
        context: ocp.checkpoint_managers.PreservationContext,
    ) -> Sequence[bool]:
        """Return whether each checkpoint must be kept for fork lineage."""
        del context
        pinned_steps = self.get_pinned_steps()
        return [checkpoint.step in pinned_steps for checkpoint in checkpoints]


def build_checkpoint_preservation_policy(
    *,
    max_to_keep: int | None,
    get_pinned_steps: Callable[[], set[int]],
) -> ocp.checkpoint_managers.PreservationPolicy:
    """Build Orbax preservation policy for ``max_to_keep`` and pinned steps.

    Args:
        max_to_keep: Maximum checkpoints to retain (`None` keeps all recent steps).
        get_pinned_steps: Callable returning pinned Orbax step identifiers.

    Returns:
        Combined preservation policy equivalent to legacy ``max_to_keep`` plus
        ``should_keep_fn`` behavior.
    """
    return ocp.checkpoint_managers.AnyPreservationPolicy(
        policies=[
            ocp.checkpoint_managers.LatestN(n=max_to_keep),
            PinnedStepsPreservationPolicy(get_pinned_steps=get_pinned_steps),
        ],
    )
