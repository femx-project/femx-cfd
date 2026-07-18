"""Reusable validation primitives for versioned CFD case formats."""

from pathlib import Path
from typing import Optional

import numpy as np

from femx_cfd._typing import Config, Dynamic, PathLike


LENGTH_SCALES = {"mm": 1.0e-3, "cm": 1.0e-2, "m": 1.0}


def mapping(value: Dynamic, name: str) -> Config:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def positive(value: Dynamic, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive number") from error
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive number")
    return number


def nonnegative(value: Dynamic, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a nonnegative number") from error
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a nonnegative number")
    return number


def integer_multiple(value: float, unit: float, name: str) -> int:
    ratio = value / unit
    count = int(round(ratio))
    if count <= 0 or not np.isclose(ratio, count, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{name} must be an integer multiple of time.dt")
    return count


def resolve_file(
    value: Dynamic,
    name: str,
    base_dir: Optional[PathLike],
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty path")
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{name} does not exist: {path}")
    return str(path)


def validate_mesh(value: Dynamic, base_dir: Optional[PathLike]) -> Config:
    if isinstance(value, str):
        value = {"path": value, "length_unit": "m"}
    options = mapping(value, "mesh")
    path = resolve_file(options.get("path"), "mesh", base_dir)
    length_unit = options.get("length_unit", "m")
    if length_unit not in LENGTH_SCALES:
        choices = ", ".join(LENGTH_SCALES)
        raise ValueError(f"mesh.length_unit must be one of: {choices}")
    return {"path": path, "length_unit": length_unit}


def validate_fluid(value: Dynamic) -> Config:
    fluid = mapping(value, "fluid")
    fluid["rho"] = positive(fluid.get("rho"), "fluid.rho")
    fluid["mu"] = positive(fluid.get("mu"), "fluid.mu")
    return fluid


def validate_solver(value: Dynamic) -> Config:
    solver = mapping({} if value is None else value, "solver")
    backend = solver.get("backend", "petsc")
    if not isinstance(backend, str) or not backend:
        raise ValueError("solver.backend must be a nonempty string")
    petsc_options = solver.get("petsc_options")
    if petsc_options is not None and not isinstance(petsc_options, str):
        raise ValueError("solver.petsc_options must be a string")
    options = solver.get("options")
    if options is not None and not isinstance(options, dict):
        raise ValueError("solver.options must be a mapping")
    solver["backend"] = backend
    return solver


def validate_fixed_boundaries(
    conditions: Dynamic,
    *,
    excluded: set[str],
) -> list[Config]:
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("boundary_conditions must be a nonempty list")

    names = set()
    for index, condition in enumerate(conditions):
        condition = mapping(condition, f"boundary_conditions[{index}]")
        boundary = condition.get("boundary")
        if not isinstance(boundary, str) or not boundary:
            raise ValueError(
                f"boundary_conditions[{index}].boundary must be a nonempty string"
            )
        if boundary in excluded:
            raise ValueError(f"controlled boundary '{boundary}' must not also be fixed")
        if boundary in names:
            raise ValueError(f"duplicate boundary condition: {boundary}")
        names.add(boundary)

        kind = condition.get("kind")
        if kind == "velocity":
            try:
                vector = np.asarray(condition.get("value"), dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"boundary_conditions[{index}].value must contain "
                    "2 or 3 finite values"
                ) from error
            if (
                vector.ndim != 1
                or vector.size not in (2, 3)
                or not np.all(np.isfinite(vector))
            ):
                raise ValueError(
                    f"boundary_conditions[{index}].value must contain "
                    "2 or 3 finite values"
                )
            condition["value"] = vector.tolist()
            continue
        if kind == "pressure":
            try:
                number = float(condition.get("value"))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"boundary_conditions[{index}].value must be finite"
                ) from error
            if not np.isfinite(number):
                raise ValueError(f"boundary_conditions[{index}].value must be finite")
            condition["value"] = number
            continue
        raise ValueError(
            f"boundary_conditions[{index}].kind must be 'velocity' or 'pressure'"
        )
    return conditions


__all__ = [
    "LENGTH_SCALES",
    "integer_multiple",
    "mapping",
    "nonnegative",
    "positive",
    "resolve_file",
    "validate_fixed_boundaries",
    "validate_fluid",
    "validate_mesh",
    "validate_solver",
]
