"""Migrations for portable forward-case documents."""

from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional

import femx

from femx_cfd._typing import Config, PathLike


Migration = Callable[[Config, Optional[PathLike]], Config]


def _mesh_dimension(case: Config, base_dir: Optional[PathLike]) -> int:
    options = case.get("mesh")
    value = options if isinstance(options, str) else None
    if isinstance(options, dict):
        value = options.get("path")
    if not isinstance(value, str) or not value:
        raise ValueError("mesh.path must be a nonempty path")
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"mesh does not exist: {path}")
    return int(femx.Mesh.read(path).dimension)


def migrate_v1_to_v2(
    case: Config,
    base_dir: Optional[PathLike] = None,
) -> Config:
    """Replace version 1 no-slip presets with explicit zero velocities."""

    migrated = deepcopy(case)
    conditions = migrated.get("boundary_conditions")
    needs_dimension = isinstance(conditions, list) and any(
        isinstance(condition, dict) and condition.get("kind") == "no_slip"
        for condition in conditions
    )
    dimension = _mesh_dimension(migrated, base_dir) if needs_dimension else None
    if isinstance(conditions, list):
        for condition in conditions:
            if isinstance(condition, dict) and condition.get("kind") == "no_slip":
                assert dimension is not None
                condition["kind"] = "velocity"
                condition["value"] = [0.0] * dimension
            elif isinstance(condition, dict) and condition.get("kind") == "pressure":
                condition.setdefault("value", 0.0)
    migrated["version"] = 2
    return migrated


MIGRATIONS: dict[int, Migration] = {1: migrate_v1_to_v2}


def migrate_forward_case(
    case: Config,
    base_dir: Optional[PathLike] = None,
) -> Config:
    """Migrate a forward document as far as the registered chain allows."""

    migrated = deepcopy(case)
    while True:
        version = migrated.get("version")
        migration = MIGRATIONS.get(version) if type(version) is int else None
        if migration is None:
            return migrated
        previous = version
        migrated = migration(migrated, base_dir)
        if migrated.get("version") == previous:
            raise RuntimeError(f"forward case migration {previous} made no progress")


__all__ = ["MIGRATIONS", "migrate_forward_case", "migrate_v1_to_v2"]
