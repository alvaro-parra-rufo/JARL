"""Pattern matching for labels, names, and other string sets."""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Iterable

__all__ = ["jarl_pattern_match", "match_patterns"]


def jarl_pattern_match(value: str, pattern: str) -> bool:
    """Return whether ``value`` matches ``pattern`` exactly or as a glob."""
    return value == pattern or fnmatch.fnmatch(value, pattern)


def match_patterns(
    values: Iterable[str],
    *,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
    matcher: Callable[[str, str], bool] | None = None,
) -> bool:
    """Return whether any of ``values`` passes include/exclude pattern filters.

    Args:
        values: Candidate strings (for example tool labels or names).
        include: Patterns that at least one value must match. ``None`` skips
            the include dimension. An empty set never matches.
        exclude: Patterns that disqualify a value when matched. ``None`` skips
            the exclude dimension. An empty set excludes nothing.
        matcher: Callable ``(value, pattern) -> bool``. Defaults to exact
            equality plus ``fnmatch`` glob semantics.

    Returns:
        ``True`` when no exclude rule hits and include rules (if any) pass.
    """
    match = matcher or jarl_pattern_match
    value_list = tuple(values)

    if exclude is not None:
        for value in value_list:
            for pattern in exclude:
                if match(value, pattern):
                    return False

    if include is None:
        return True

    if not include:
        return False

    for value in value_list:
        for pattern in include:
            if match(value, pattern):
                return True
    return False
