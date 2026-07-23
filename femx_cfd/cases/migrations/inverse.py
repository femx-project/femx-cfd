"""Migrations for portable inverse-case documents."""

from copy import deepcopy
from typing import Callable, Optional

from femx_cfd._typing import Config, PathLike


_Migration = Callable[[Config, Optional[PathLike]], Config]


def _migrate_v1_to_v2(
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


def _migrate_v2_to_v3(
    case: Config,
    base_dir: Optional[PathLike] = None,
) -> Config:
    """Adopt the output configuration while preserving earlier defaults."""

    del base_dir
    migrated = deepcopy(case)
    if "output" in migrated:
        raise ValueError("output requires case.version 3")
    migrated["version"] = 3
    return migrated


_MIGRATIONS: dict[int, _Migration] = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
}


def migrate_inverse_case(
    case: Config,
    base_dir: Optional[PathLike] = None,
) -> Config:
    """Migrate a supported inverse document to the current version."""

    migrated = deepcopy(case)
    while True:
        version = migrated.get("version")
        migration = _MIGRATIONS.get(version) if type(version) is int else None
        if migration is None:
            return migrated
        previous = version
        migrated = migration(migrated, base_dir)
        if migrated.get("version") == previous:
            raise RuntimeError(f"inverse case migration {previous} made no progress")


__all__ = [
    "migrate_inverse_case",
]
