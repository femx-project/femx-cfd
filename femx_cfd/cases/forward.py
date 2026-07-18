"""Forward case format version 1 validation and path resolution."""

from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from femx_cfd._typing import Config, Dynamic, PathLike
from femx_cfd.cases.validation import LENGTH_SCALES, validate_solver


CASE_FORMAT = "femx-cfd-forward"
CASE_VERSION = 1


def _mapping(value: Dynamic, name: str) -> Config:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive(value: Dynamic, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive number") from error
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive number")
    return number


def _multiple(value: float, unit: float, name: str) -> int:
    ratio = value / unit
    count = int(round(ratio))
    if count <= 0 or not np.isclose(ratio, count, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{name} must be an integer multiple of time.dt")
    return count


def _validate_inlet(inlet: Config, dt: float, end_time: float) -> None:
    boundary = inlet.get("boundary")
    if not isinstance(boundary, str) or not boundary:
        raise ValueError("inlet.boundary must be a nonempty string")

    normal = inlet.get("normal", "auto")
    if not isinstance(normal, str):
        try:
            normal = [float(value) for value in normal]
        except (TypeError, ValueError) as error:
            raise ValueError("inlet.normal must be 'auto' or a vector") from error
    elif normal != "auto":
        raise ValueError("inlet.normal must be 'auto' or a vector")

    profile = _mapping(inlet.get("profile"), "inlet.profile")
    profile_kind = profile.get("kind")
    if profile_kind == "uniform":
        _positive(profile.get("value"), "inlet.profile.value")
    elif profile_kind == "poisson":
        _positive(profile.get("peak"), "inlet.profile.peak")
    else:
        raise ValueError("inlet.profile.kind must be 'uniform' or 'poisson'")

    waveform = _mapping(
        inlet.get("waveform", {"kind": "uniform", "value": 1.0}),
        "inlet.waveform",
    )
    waveform_kind = waveform.get("kind", "uniform")
    if waveform_kind not in ("uniform", "harmonics"):
        raise ValueError("inlet.waveform.kind must be 'uniform' or 'harmonics'")

    period = _positive(inlet.get("period", end_time), "inlet.period")
    _multiple(period, dt, "inlet.period")
    ratio = end_time / period
    cycles = int(round(ratio))
    if cycles <= 0 or not np.isclose(ratio, cycles, rtol=0.0, atol=1.0e-10):
        raise ValueError("time.end_time must be an integer multiple of inlet.period")

    inlet["normal"] = normal
    inlet["period"] = period
    inlet["waveform"] = waveform


def _validate_boundary_conditions(
    conditions: Dynamic,
    inlet_boundary: str,
) -> None:
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("boundary_conditions must be a nonempty list")

    names = set()
    for index, condition in enumerate(conditions):
        condition = _mapping(condition, f"boundary_conditions[{index}]")
        boundary = condition.get("boundary")
        if not isinstance(boundary, str) or not boundary:
            raise ValueError(
                f"boundary_conditions[{index}].boundary must be a nonempty string"
            )
        if boundary == inlet_boundary:
            raise ValueError("the inlet must not also be a fixed boundary condition")
        if boundary in names:
            raise ValueError(f"duplicate boundary condition: {boundary}")
        names.add(boundary)

        kind = condition.get("kind")
        if kind == "no_slip":
            continue
        if kind == "pressure":
            try:
                value = float(condition.get("value", 0.0))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"boundary_conditions[{index}].value must be finite"
                ) from error
            if not np.isfinite(value):
                raise ValueError(f"boundary_conditions[{index}].value must be finite")
            condition["value"] = value
            continue
        raise ValueError(
            f"boundary_conditions[{index}].kind must be 'no_slip' or 'pressure'"
        )


def validate_forward_case(
    case: Dynamic,
    base_dir: Optional[PathLike] = None,
) -> Config:
    """Validate and normalize a plain forward-case mapping."""

    case = deepcopy(_mapping(case, "case"))
    if case.get("format") != CASE_FORMAT:
        raise ValueError(f"case.format must be '{CASE_FORMAT}'")
    if case.get("version") != CASE_VERSION:
        raise ValueError(f"case.version must be {CASE_VERSION}")

    mesh_options = case.get("mesh")
    if isinstance(mesh_options, str):
        mesh_options = {"path": mesh_options, "length_unit": "m"}
    mesh_options = _mapping(mesh_options, "mesh")
    mesh_value = mesh_options.get("path")
    if not isinstance(mesh_value, str) or not mesh_value:
        raise ValueError("mesh.path must be a nonempty path")
    mesh = Path(mesh_value).expanduser()
    if not mesh.is_absolute() and base_dir is not None:
        mesh = Path(base_dir) / mesh
    mesh = mesh.resolve()
    if not mesh.is_file():
        raise ValueError(f"mesh does not exist: {mesh}")
    length_unit = mesh_options.get("length_unit", "m")
    if length_unit not in LENGTH_SCALES:
        choices = ", ".join(LENGTH_SCALES)
        raise ValueError(f"mesh.length_unit must be one of: {choices}")
    case["mesh"] = {"path": str(mesh), "length_unit": length_unit}

    fluid = _mapping(case.get("fluid"), "fluid")
    fluid["rho"] = _positive(fluid.get("rho"), "fluid.rho")
    fluid["mu"] = _positive(fluid.get("mu"), "fluid.mu")

    time = _mapping(case.get("time"), "time")
    time["dt"] = _positive(time.get("dt"), "time.dt")
    time["end_time"] = _positive(time.get("end_time"), "time.end_time")
    _multiple(time["end_time"], time["dt"], "time.end_time")

    inlet = _mapping(case.get("inlet"), "inlet")
    _validate_inlet(inlet, time["dt"], time["end_time"])
    _validate_boundary_conditions(case.get("boundary_conditions"), inlet["boundary"])

    case["solver"] = validate_solver(case.get("solver"))
    return case


def load_forward_case(path: PathLike) -> Config:
    """Load a YAML case, resolving its mesh relative to the YAML file."""

    case_path = Path(path).expanduser().resolve()
    if not case_path.is_file():
        raise ValueError(f"case does not exist: {case_path}")
    loaded = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    return validate_forward_case(loaded, base_dir=case_path.parent)


__all__ = [
    "CASE_FORMAT",
    "CASE_VERSION",
    "LENGTH_SCALES",
    "load_forward_case",
    "validate_forward_case",
]
