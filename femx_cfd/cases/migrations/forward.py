"""Migrations for portable forward-case documents."""

from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional

import femx

from femx_cfd._typing import Config, PathLike


_Migration = Callable[[Config, Optional[PathLike]], Config]


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


def _migrate_v1_to_v2(
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


def _migrate_v2_to_v3(
    case: Config,
    base_dir: Optional[PathLike] = None,
) -> Config:
    """Adopt unified boundaries and output while preserving earlier defaults."""

    del base_dir
    migrated = deepcopy(case)
    if "output" in migrated:
        raise ValueError("output requires case.version 3")
    inlet = migrated.pop("inlet", None)
    conditions = migrated.setdefault("boundary_conditions", [])
    if isinstance(conditions, list):
        for condition in conditions:
            if isinstance(condition, dict):
                condition.setdefault("source", "analytic")
    if isinstance(inlet, dict):
        condition = {"kind": "velocity", **inlet}
        condition.setdefault("source", "analytic")
        if isinstance(conditions, list):
            conditions.insert(0, condition)
    migrated["version"] = 3
    return migrated


_MIGRATIONS: dict[int, _Migration] = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
}


def migrate_forward_case(
    case: Config,
    base_dir: Optional[PathLike] = None,
) -> Config:
    """Migrate a forward document as far as the registered chain allows."""

    migrated = deepcopy(case)
    while True:
        version = migrated.get("version")
        migration = _MIGRATIONS.get(version) if type(version) is int else None
        if migration is None:
            return migrated
        previous = version
        migrated = migration(migrated, base_dir)
        if migrated.get("version") == previous:
            raise RuntimeError(f"forward case migration {previous} made no progress")


__all__ = [
    "migrate_forward_case",
]
