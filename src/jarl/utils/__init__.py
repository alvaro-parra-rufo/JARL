"""General-purpose utilities for the jarl package."""

from jarl.utils.dicts import deep_merge_dicts, flatten_dict, unflatten_dict
from jarl.utils.env import find_project_root, load_project_env
from jarl.utils.extras import OptionalExtra, is_extra_available
from jarl.utils.hash import dict_hash
from jarl.utils.ids import short_uuid
from jarl.utils.io import write_text_atomic
from jarl.utils.pattern_filter import jarl_pattern_match, match_patterns

__all__ = [
    "OptionalExtra",
    "deep_merge_dicts",
    "dict_hash",
    "find_project_root",
    "flatten_dict",
    "is_extra_available",
    "jarl_pattern_match",
    "load_project_env",
    "match_patterns",
    "short_uuid",
    "unflatten_dict",
    "write_text_atomic",
]
