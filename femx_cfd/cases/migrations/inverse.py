"""Migrations for portable inverse-case documents."""

from copy import deepcopy
from typing import Optional

from femx_cfd._typing import Config, PathLike


def migrate_v1_to_v2(
    case: Config,
    base_dir: Optional[PathLike] = None,
) -> Config:
    """Preserve the version 1 normal-control default explicitly."""

    del base_dir
    migrated = deepcopy(case)
    control = migrated.get("control")
    if isinstance(control, dict):
        control.setdefault("kind", "normal")
    migrated["version"] = 2
    return migrated


def migrate_inverse_case(
    case: Config,
    base_dir: Optional[PathLike] = None,
) -> Config:
    """Migrate a supported inverse document to the current version."""

    migrated = deepcopy(case)
    if migrated.get("version") == 1:
        migrated = migrate_v1_to_v2(migrated, base_dir)
    return migrated


__all__ = ["migrate_inverse_case", "migrate_v1_to_v2"]
