"""Hashing helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["dict_hash"]


def dict_hash(d: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of a dictionary.

    Keys are sorted recursively so insertion order doesn't affect the hash.

    Args:
        d: Dictionary to hash.

    Returns:
        Hex digest string.
    """
    serialized = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()
