"""One-way migrations for supported portable case schemas."""

from .forward import migrate_forward_case
from .forward import migrate_v1_to_v2 as migrate_forward_v1_to_v2
from .inverse import migrate_inverse_case
from .inverse import migrate_v1_to_v2 as migrate_inverse_v1_to_v2

migrate_v1_to_v2 = migrate_forward_v1_to_v2

__all__ = [
    "migrate_forward_case",
    "migrate_forward_v1_to_v2",
    "migrate_inverse_case",
    "migrate_inverse_v1_to_v2",
    "migrate_v1_to_v2",
]
