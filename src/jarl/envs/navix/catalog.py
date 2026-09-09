"""Navix map metadata and transfer-compatibility registry."""

from __future__ import annotations

from dataclasses import dataclass

from jarl.envs.navix.aliases import canonical_navix_env_id, canonical_navix_env_ids

__all__ = [
    "NAVIX_DISCRETE_ACTION_NAMES",
    "NavixMapContract",
    "assert_transfer_compatible",
    "get_contract",
    "list_registered_maps",
    "register_map",
]

_REGISTRY: dict[str, NavixMapContract] = {}

NAVIX_DISCRETE_ACTION_NAMES: tuple[str, ...] = (
    "rotate_ccw",
    "rotate_cw",
    "forward",
    "pickup",
    "drop",
    "toggle",
    "done",
)
"""MiniGrid-style Navix action names (indices 0-6)."""

# Transfer group for canonical maps sharing symbolic_first_person obs + discrete actions.
# Edit this label when registering a catalog slice with a different contract.
_BUILTIN_TRANSFER_GROUP = "navix_symbolic_fp_147_7"
_REFERENCE_ENV_ID = "Navix-Empty-5x5-v0"


@dataclass(frozen=True, slots=True)
class NavixMapContract:
    """Transfer metadata for one Navix environment id.

    Args:
        env_id: Gymnasium environment identifier.
        processed_obs_shape: Observation shape after ``preprocess_observation``.
        action_names: Ordered discrete action labels.
        transfer_group: Compatibility group for checkpoint transfer.
    """

    env_id: str
    processed_obs_shape: tuple[int, ...]
    action_names: tuple[str, ...]
    transfer_group: str


def register_map(contract: NavixMapContract) -> None:
    """Register or replace a Navix map contract.

    Args:
        contract: Map metadata to store in the in-memory registry.
    """
    _REGISTRY[contract.env_id] = contract


def get_contract(env_id: str) -> NavixMapContract:
    """Return the registered contract for ``env_id``.

    Args:
        env_id: Gymnasium environment identifier.

    Returns:
        Registered ``NavixMapContract``.

    Raises:
        KeyError: If ``env_id`` is not registered.
    """
    canonical_id = canonical_navix_env_id(env_id)
    try:
        return _REGISTRY[canonical_id]
    except KeyError as exc:
        msg = f"Unknown Navix map env_id: {env_id!r}. Register it with register_map()."
        raise KeyError(msg) from exc


def list_registered_maps() -> tuple[NavixMapContract, ...]:
    """Return all registered map contracts sorted by ``env_id``."""
    return tuple(_REGISTRY[env_id] for env_id in sorted(_REGISTRY))


def assert_transfer_compatible(source_env_id: str, target_env_id: str) -> None:
    """Validate that two Navix maps can exchange checkpoints.

    Args:
        source_env_id: Source environment identifier.
        target_env_id: Target environment identifier.

    Raises:
        ValueError: If the maps belong to different transfer groups or contracts differ.
    """
    source = get_contract(source_env_id)
    target = get_contract(target_env_id)
    if source.transfer_group != target.transfer_group:
        msg = (
            f"Incompatible Navix transfer groups: {source_env_id!r} "
            f"({source.transfer_group!r}) -> {target_env_id!r} ({target.transfer_group!r})."
        )
        raise ValueError(msg)
    if source.processed_obs_shape != target.processed_obs_shape:
        msg = (
            f"Incompatible Navix observation shapes: {source_env_id!r} "
            f"{source.processed_obs_shape} -> {target_env_id!r} {target.processed_obs_shape}."
        )
        raise ValueError(msg)
    if source.action_names != target.action_names:
        msg = f"Incompatible Navix action spaces: {source_env_id!r} -> {target_env_id!r}."
        raise ValueError(msg)


def _probe_contract(env_id: str, transfer_group: str) -> NavixMapContract:
    from jarl.envs.navix.full_jit import make_navix_full_jit_env

    env = make_navix_full_jit_env(env_id)
    return NavixMapContract(
        env_id=env_id,
        processed_obs_shape=env.get_processed_observation_shape(),
        action_names=env.discrete_action_names(),
        transfer_group=transfer_group,
    )


def _register_builtin_maps() -> None:
    reference = _probe_contract(_REFERENCE_ENV_ID, _BUILTIN_TRANSFER_GROUP)
    for env_id in canonical_navix_env_ids():
        register_map(
            NavixMapContract(
                env_id=env_id,
                processed_obs_shape=reference.processed_obs_shape,
                action_names=reference.action_names,
                transfer_group=reference.transfer_group,
            )
        )


_register_builtin_maps()
