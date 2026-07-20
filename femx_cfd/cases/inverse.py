"""Inverse-analysis case format version 1 validation and path resolution."""

from copy import deepcopy
from pathlib import Path
from typing import Optional

import yaml

from femx_cfd._typing import Config, Dynamic, PathLike

from .validation import (
    integer_multiple,
    mapping,
    nonnegative,
    positive,
    resolve_file,
    validate_fixed_boundaries,
    validate_fluid,
    validate_mesh,
    validate_solver,
)


CASE_FORMAT = "femx-cfd-inverse"
CASE_VERSION = 1
METHODS = {"var", "envar"}


def _positive_integer(value: Dynamic, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if number <= 0 or number != value:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _validate_control(value: Dynamic) -> Config:
    control = mapping(value, "control")
    boundary = control.get("boundary")
    if not isinstance(boundary, str) or not boundary:
        raise ValueError("control.boundary must be a nonempty string")
    if control.get("kind", "normal") != "normal":
        raise ValueError("control.kind must be 'normal'")
    normal = control.get("normal", "auto")
    if not isinstance(normal, str):
        try:
            normal = [float(component) for component in normal]
        except (TypeError, ValueError) as error:
            raise ValueError("control.normal must be 'auto' or a vector") from error
    elif normal != "auto":
        raise ValueError("control.normal must be 'auto' or a vector")
    control["kind"] = "normal"
    control["normal"] = normal
    return control


def _validate_time(value: Dynamic) -> Config:
    time = mapping(value, "time")
    period = positive(time.get("period"), "time.period")
    dt = positive(time.get("dt"), "time.dt")
    end_time = positive(time.get("end_time", period), "time.end_time")
    integer_multiple(period, dt, "time.period")
    integer_multiple(end_time, dt, "time.end_time")
    time.update({"period": period, "dt": dt, "end_time": end_time})
    return time


def _validate_observations(value: Dynamic, base_dir: Optional[PathLike]) -> Config:
    observations = mapping(value, "observations")
    if observations.get("kind", "velocity") != "velocity":
        raise ValueError("observations.kind must be 'velocity'")
    observations["kind"] = "velocity"
    observations["path"] = resolve_file(
        observations.get("path"),
        "observations.path",
        base_dir,
    )
    return observations


def _validate_initialization(
    value: Dynamic,
    base_dir: Optional[PathLike],
) -> Config:
    initialization = mapping(value, "initialization")
    inlet = mapping(initialization.get("inlet"), "initialization.inlet")
    source = inlet.get("source", "file")
    if source == "file":
        inlet["path"] = resolve_file(
            inlet.get("path"),
            "initialization.inlet.path",
            base_dir,
        )
    elif source == "observations":
        projection = mapping(
            inlet.get("projection"),
            "initialization.inlet.projection",
        )
        projection["depth"] = positive(
            projection.get("depth"),
            "initialization.inlet.projection.depth",
        )
        projection["axial_decay"] = positive(
            projection.get("axial_decay"),
            "initialization.inlet.projection.axial_decay",
        )
        projection["laplacian_modes"] = _positive_integer(
            projection.get("laplacian_modes"),
            "initialization.inlet.projection.laplacian_modes",
        )
        projection["ridge"] = nonnegative(
            projection.get("ridge", 0.0),
            "initialization.inlet.projection.ridge",
        )
        projection["temporal_weight"] = nonnegative(
            projection.get("temporal_weight", 0.0),
            "initialization.inlet.projection.temporal_weight",
        )
    else:
        raise ValueError("initialization.inlet.source must be 'file' or 'observations'")
    inlet["source"] = source
    initialization["inlet"] = inlet
    return initialization


def _validate_var(value: Dynamic) -> Config:
    algorithm = mapping(value, "algorithm")
    optimizer = algorithm.get("optimizer", "tao")
    if optimizer not in ("scipy", "tao"):
        raise ValueError("algorithm.optimizer must be 'scipy' or 'tao'")
    algorithm["optimizer"] = optimizer
    algorithm["max_iterations"] = _positive_integer(
        algorithm.get("max_iterations", 40),
        "algorithm.max_iterations",
    )
    algorithm["gradient_tolerance"] = positive(
        algorithm.get("gradient_tolerance", 1.0e-3),
        "algorithm.gradient_tolerance",
    )
    algorithm["function_tolerance"] = nonnegative(
        algorithm.get("function_tolerance", 1.0e-12),
        "algorithm.function_tolerance",
    )
    bounds = algorithm.get("control_bounds")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("algorithm.control_bounds must contain two numbers")
    lower = nonnegative(bounds[0], "algorithm.control_bounds[0]")
    upper = positive(bounds[1], "algorithm.control_bounds[1]")
    if lower >= upper:
        raise ValueError("algorithm.control_bounds must be increasing")
    algorithm["control_bounds"] = [lower, upper]
    regularization = mapping(
        algorithm.get("regularization", {}),
        "algorithm.regularization",
    )
    for name in ("background", "space", "time"):
        regularization[name] = nonnegative(
            regularization.get(name, 0.0),
            f"algorithm.regularization.{name}",
        )
    algorithm["regularization"] = regularization
    return algorithm


def _validate_envar(value: Dynamic, ensemble_value: Dynamic) -> tuple[Config, Config]:
    algorithm = mapping(value, "algorithm")
    algorithm["prior_weight"] = nonnegative(
        algorithm.get("prior_weight", 1.0),
        "algorithm.prior_weight",
    )
    ensemble = mapping(ensemble_value, "ensemble")
    ensemble["workers"] = _positive_integer(
        ensemble.get("workers", 1),
        "ensemble.workers",
    )
    scales = mapping(ensemble.get("mode_scales"), "ensemble.mode_scales")
    if not scales:
        raise ValueError("ensemble.mode_scales must not be empty")
    for name, value in scales.items():
        scales[name] = positive(value, f"ensemble.mode_scales.{name}")
    ensemble["mode_scales"] = scales
    return algorithm, ensemble


def validate_inverse_case(
    case: Dynamic,
    base_dir: Optional[PathLike] = None,
) -> Config:
    """Validate and normalize a plain inverse-case mapping."""

    case = deepcopy(mapping(case, "case"))
    if case.get("format") != CASE_FORMAT:
        raise ValueError(f"case.format must be '{CASE_FORMAT}'")
    if case.get("version") != CASE_VERSION:
        raise ValueError(f"case.version must be {CASE_VERSION}")
    method = case.get("method")
    if method not in METHODS:
        choices = ", ".join(sorted(METHODS))
        raise ValueError(f"case.method must be one of: {choices}")

    case["mesh"] = validate_mesh(case.get("mesh"), base_dir)
    case["fluid"] = validate_fluid(case.get("fluid"))
    case["time"] = _validate_time(case.get("time"))
    case["control"] = _validate_control(case.get("control"))
    case["boundary_conditions"] = validate_fixed_boundaries(
        case.get("boundary_conditions"),
        excluded={case["control"]["boundary"]},
    )
    case["observations"] = _validate_observations(
        case.get("observations"),
        base_dir,
    )
    case["initialization"] = _validate_initialization(
        case.get("initialization"),
        base_dir,
    )
    case["solver"] = validate_solver(case.get("solver"))
    if method == "var":
        case["algorithm"] = _validate_var(case.get("algorithm"))
        case.pop("ensemble", None)
    else:
        case["algorithm"], case["ensemble"] = _validate_envar(
            case.get("algorithm"),
            case.get("ensemble"),
        )
    return case


def load_inverse_case(path: PathLike) -> Config:
    """Load a YAML inverse case and resolve its referenced inputs."""

    case_path = Path(path).expanduser().resolve()
    if not case_path.is_file():
        raise ValueError(f"case does not exist: {case_path}")
    loaded = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    return validate_inverse_case(loaded, base_dir=case_path.parent)


__all__ = [
    "CASE_FORMAT",
    "CASE_VERSION",
    "METHODS",
    "load_inverse_case",
    "validate_inverse_case",
]
