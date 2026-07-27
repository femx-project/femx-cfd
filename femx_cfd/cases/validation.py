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


def reject_unknown(value: Config, allowed: set[str], name: str) -> None:
    unknown = set(value).difference(allowed)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{name} contains unknown options: {names}")


def finite(value: Dynamic, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


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


def positive_integer(value: Dynamic, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if number <= 0 or number != value:
        raise ValueError(f"{name} must be a positive integer")
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
    reject_unknown(options, {"path", "length_unit"}, "mesh")
    path = resolve_file(options.get("path"), "mesh", base_dir)
    length_unit = options.get("length_unit", "m")
    if length_unit not in LENGTH_SCALES:
        choices = ", ".join(LENGTH_SCALES)
        raise ValueError(f"mesh.length_unit must be one of: {choices}")
    return {"path": path, "length_unit": length_unit}


def validate_fluid(value: Dynamic) -> Config:
    fluid = mapping(value, "fluid")
    reject_unknown(fluid, {"rho", "mu"}, "fluid")
    fluid["rho"] = positive(fluid.get("rho"), "fluid.rho")
    fluid["mu"] = positive(fluid.get("mu"), "fluid.mu")
    return fluid


def validate_solver(value: Dynamic) -> Config:
    solver = mapping({} if value is None else value, "solver")
    reject_unknown(
        solver,
        {"platform", "backend", "petsc_options", "options"},
        "solver",
    )
    platform = solver.get("platform", "cpu")
    if platform not in ("cpu", "cuda"):
        raise ValueError("solver.platform must be 'cpu' or 'cuda'")
    backend = solver.get("backend", "petsc")
    if backend not in ("dense", "resolve", "petsc"):
        raise ValueError("solver.backend must be 'dense', 'resolve', or 'petsc'")
    petsc_options = solver.get("petsc_options")
    if petsc_options is not None and not isinstance(petsc_options, str):
        raise ValueError("solver.petsc_options must be a string")
    options = solver.get("options")
    if options is not None and not isinstance(options, dict):
        raise ValueError("solver.options must be a mapping")
    solver.update({"platform": platform, "backend": backend})
    return solver


def validate_profile(value: Dynamic, name: str) -> Config:
    """Validate one analytic inlet spatial profile."""

    profile = mapping(value, name)
    kind = profile.get("kind")
    if kind == "uniform":
        reject_unknown(profile, {"kind", "value"}, name)
        profile["value"] = positive(profile.get("value"), f"{name}.value")
    elif kind == "poisson":
        reject_unknown(profile, {"kind", "peak", "skew"}, name)
        profile["peak"] = positive(profile.get("peak"), f"{name}.peak")
        if "skew" in profile:
            profile["skew"] = finite(profile["skew"], f"{name}.skew")
    else:
        raise ValueError(f"{name}.kind must be 'uniform' or 'poisson'")
    return profile


def validate_waveform(value: Dynamic, name: str) -> Config:
    """Validate one analytic inlet temporal waveform."""

    waveform = mapping(value, name)
    kind = waveform.get("kind", "uniform")
    if kind == "uniform":
        reject_unknown(waveform, {"kind", "value"}, name)
        waveform["value"] = finite(waveform.get("value", 1.0), f"{name}.value")
    elif kind == "harmonics":
        reject_unknown(waveform, {"kind", "mean", "terms"}, name)
        waveform["mean"] = finite(waveform.get("mean", 1.0), f"{name}.mean")
        terms = waveform.get("terms", [])
        if not isinstance(terms, list):
            raise ValueError(f"{name}.terms must be a list")
        for index, term in enumerate(terms):
            prefix = f"{name}.terms[{index}]"
            term = mapping(term, prefix)
            reject_unknown(term, {"order", "amplitude", "phase", "kind"}, prefix)
            term["order"] = positive_integer(term.get("order"), f"{prefix}.order")
            term["amplitude"] = finite(
                term.get("amplitude"),
                f"{prefix}.amplitude",
            )
            term["phase"] = finite(term.get("phase", 0.0), f"{prefix}.phase")
            term_kind = term.get("kind", "sin")
            if term_kind not in ("sin", "cos"):
                raise ValueError(f"{prefix}.kind must be 'sin' or 'cos'")
            term["kind"] = term_kind
            terms[index] = term
        waveform["terms"] = terms
    else:
        raise ValueError(f"{name}.kind must be 'uniform' or 'harmonics'")
    waveform["kind"] = kind
    return waveform


def validate_output(
    value: Dynamic,
    base_dir: Optional[PathLike],
    *,
    num_steps: int,
) -> Config:
    """Validate output placement and trajectory sampling options."""

    output = mapping({} if value is None else value, "output")
    reject_unknown(output, {"directory", "every"}, "output")

    every = output.get("every", 1)
    if isinstance(every, bool) or not isinstance(every, int) or every <= 0:
        raise ValueError("output.every must be a positive integer")
    if num_steps % every != 0:
        raise ValueError(
            f"output.every ({every}) must divide the {num_steps} integration steps"
        )

    validated = {"every": every}
    directory = output.get("directory")
    if directory is not None:
        if not isinstance(directory, str) or not directory:
            raise ValueError("output.directory must be a nonempty path")
        path = Path(directory).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        validated["directory"] = str(path.resolve())
    return validated


def validate_fixed_boundary(condition: Dynamic, name: str) -> Config:
    """Validate one uniform velocity or pressure boundary condition."""

    condition = mapping(condition, name)
    reject_unknown(condition, {"boundary", "kind", "source", "value"}, name)
    boundary = condition.get("boundary")
    if not isinstance(boundary, str) or not boundary:
        raise ValueError(f"{name}.boundary must be a nonempty string")
    source = condition.get("source")
    if source is not None and source != "analytic":
        raise ValueError(f"{name}.source must be 'analytic' for a fixed value")

    kind = condition.get("kind")
    if kind == "velocity":
        try:
            vector = np.asarray(condition.get("value"), dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{name}.value must contain 2 or 3 finite values"
            ) from error
        if (
            vector.ndim != 1
            or vector.size not in (2, 3)
            or not np.all(np.isfinite(vector))
        ):
            raise ValueError(f"{name}.value must contain 2 or 3 finite values")
        condition["value"] = vector.tolist()
        return condition
    if kind == "pressure":
        try:
            number = float(condition.get("value"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name}.value must be finite") from error
        if not np.isfinite(number):
            raise ValueError(f"{name}.value must be finite")
        condition["value"] = number
        return condition
    raise ValueError(f"{name}.kind must be 'velocity' or 'pressure'")


def validate_fixed_boundaries(
    conditions: Dynamic,
    *,
    excluded: set[str],
) -> list[Config]:
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("boundary_conditions must be a nonempty list")

    names = set()
    for index, condition in enumerate(conditions):
        condition = validate_fixed_boundary(
            condition,
            f"boundary_conditions[{index}]",
        )
        boundary = condition["boundary"]
        if boundary in excluded:
            raise ValueError(f"controlled boundary '{boundary}' must not also be fixed")
        if boundary in names:
            raise ValueError(f"duplicate boundary condition: {boundary}")
        names.add(boundary)
        conditions[index] = condition
    return conditions


__all__ = [
    "LENGTH_SCALES",
    "finite",
    "integer_multiple",
    "mapping",
    "nonnegative",
    "positive",
    "positive_integer",
    "reject_unknown",
    "resolve_file",
    "validate_fixed_boundaries",
    "validate_fixed_boundary",
    "validate_fluid",
    "validate_mesh",
    "validate_output",
    "validate_profile",
    "validate_solver",
    "validate_waveform",
]
