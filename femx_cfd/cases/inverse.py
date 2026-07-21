"""Inverse-analysis case validation, migration, and path resolution."""

from copy import deepcopy
import math
from pathlib import Path
from typing import Optional

import yaml

from femx_cfd._typing import Config, Dynamic, PathLike
from femx_cfd.cases.migrations import migrate_inverse_case

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
CASE_VERSION = 2
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


def _nonnegative_integer(value: Dynamic, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a nonnegative integer") from error
    if number < 0 or number != value:
        raise ValueError(f"{name} must be a nonnegative integer")
    return number


def _validate_control(value: Dynamic) -> Config:
    control = mapping(value, "control")
    boundary = control.get("boundary")
    if not isinstance(boundary, str) or not boundary:
        raise ValueError("control.boundary must be a nonempty string")
    kind = control.get("kind", "vector")
    if kind not in ("vector", "normal"):
        raise ValueError("control.kind must be 'vector' or 'normal'")
    control["kind"] = kind
    if kind == "vector":
        if "normal" in control:
            raise ValueError("control.normal is only valid for normal control")
        return control

    normal = control.get("normal", "auto")
    if not isinstance(normal, str):
        try:
            normal = [float(component) for component in normal]
        except (TypeError, ValueError) as error:
            raise ValueError("control.normal must be 'auto' or a vector") from error
    elif normal != "auto":
        raise ValueError("control.normal must be 'auto' or a vector")
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
    time_weights = observations.get("time_weights")
    if time_weights is not None:
        if not isinstance(time_weights, list) or not time_weights:
            raise ValueError(
                "observations.time_weights must be a nonempty list"
            )
        observations["time_weights"] = [
            positive(weight, f"observations.time_weights[{index}]")
            for index, weight in enumerate(time_weights)
        ]
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


def _validate_var(value: Dynamic, control_kind: str) -> Config:
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
    if control_kind == "normal":
        lower = nonnegative(bounds[0], "algorithm.control_bounds[0]")
        upper = positive(bounds[1], "algorithm.control_bounds[1]")
    else:
        try:
            lower, upper = (float(bound) for bound in bounds)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "algorithm.control_bounds must contain two numbers"
            ) from error
        if not all(map(math.isfinite, (lower, upper))):
            raise ValueError("algorithm.control_bounds must contain two finite numbers")
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


def _legacy_modes(scales: Config) -> Config:
    """Translate the former fixed mode names into configurable definitions."""
    definitions = {
        "flow_mean": ("uniform", "mean"),
        "flow_sin1": ("uniform", "sin1"),
        "flow_cos1": ("uniform", "cos1"),
        "flow_sin2": ("uniform", "sin2"),
        "flow_cos2": ("uniform", "cos2"),
        "skew_mean": ("skew", "mean"),
        "skew_sin1": ("skew", "sin1"),
        "skew_cos1": ("skew", "cos1"),
        "shape_mean": ("peaking", "mean"),
        "shape_cos2": ("peaking", "cos2"),
    }
    unknown = set(scales).difference(definitions)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"ensemble.mode_scales contains unknown modes: {names}")

    spatial = {"uniform": {"kind": "uniform"}}
    temporal = {"mean": {"kind": "constant"}}
    combined = []
    for name, scale in scales.items():
        spatial_name, temporal_name = definitions[name]
        if spatial_name == "skew":
            spatial[spatial_name] = {"kind": "surface_laplacian", "index": 1}
        elif spatial_name == "peaking":
            spatial[spatial_name] = {"kind": "surface_laplacian", "index": 0}
        if temporal_name != "mean":
            kind = "sine" if temporal_name.startswith("sin") else "cosine"
            temporal[temporal_name] = {
                "kind": kind,
                "order": int(temporal_name[-1]),
            }
        combined.append(
            {
                "name": name,
                "spatial": spatial_name,
                "temporal": temporal_name,
                "scale": scale,
            }
        )
    return {"spatial": spatial, "temporal": temporal, "combined": combined}


def _validate_ensemble_modes(value: Dynamic) -> Config:
    modes = mapping(value, "ensemble.modes")
    spatial = mapping(modes.get("spatial"), "ensemble.modes.spatial")
    temporal = mapping(modes.get("temporal"), "ensemble.modes.temporal")
    combined = modes.get("combined")
    if not spatial:
        raise ValueError("ensemble.modes.spatial must not be empty")
    if not temporal:
        raise ValueError("ensemble.modes.temporal must not be empty")
    if not isinstance(combined, list) or not combined:
        raise ValueError("ensemble.modes.combined must be a nonempty list")

    for name, definition in spatial.items():
        if not isinstance(name, str) or not name:
            raise ValueError("ensemble.modes.spatial names must be nonempty strings")
        definition = mapping(definition, f"ensemble.modes.spatial.{name}")
        kind = definition.get("kind")
        if kind == "uniform":
            pass
        elif kind == "surface_laplacian":
            definition["index"] = _nonnegative_integer(
                definition.get("index"),
                f"ensemble.modes.spatial.{name}.index",
            )
        else:
            raise ValueError(
                f"ensemble.modes.spatial.{name}.kind must be 'uniform' or "
                "'surface_laplacian'"
            )
        definition["kind"] = kind
        spatial[name] = definition

    for name, definition in temporal.items():
        if not isinstance(name, str) or not name:
            raise ValueError("ensemble.modes.temporal names must be nonempty strings")
        definition = mapping(definition, f"ensemble.modes.temporal.{name}")
        kind = definition.get("kind")
        if kind == "constant":
            pass
        elif kind in ("sine", "cosine"):
            definition["order"] = _positive_integer(
                definition.get("order"),
                f"ensemble.modes.temporal.{name}.order",
            )
            phase = definition.get("phase", 0.0)
            try:
                phase = float(phase)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"ensemble.modes.temporal.{name}.phase must be finite"
                ) from error
            if not math.isfinite(phase):
                raise ValueError(f"ensemble.modes.temporal.{name}.phase must be finite")
            definition["phase"] = phase
        else:
            raise ValueError(
                f"ensemble.modes.temporal.{name}.kind must be 'constant', "
                "'sine', or 'cosine'"
            )
        definition["kind"] = kind
        temporal[name] = definition

    names = set()
    normalized = []
    for index, definition in enumerate(combined):
        prefix = f"ensemble.modes.combined[{index}]"
        definition = mapping(definition, prefix)
        name = definition.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{prefix}.name must be a nonempty string")
        if name in names:
            raise ValueError(f"ensemble.modes.combined names must be unique: {name}")
        names.add(name)
        spatial_name = definition.get("spatial")
        temporal_name = definition.get("temporal")
        if not isinstance(spatial_name, str) or spatial_name not in spatial:
            raise ValueError(f"{prefix}.spatial must name a spatial mode")
        if not isinstance(temporal_name, str) or temporal_name not in temporal:
            raise ValueError(f"{prefix}.temporal must name a temporal mode")
        normalized.append(
            {
                "name": name,
                "spatial": spatial_name,
                "temporal": temporal_name,
                "scale": positive(definition.get("scale"), f"{prefix}.scale"),
            }
        )
    return {"spatial": spatial, "temporal": temporal, "combined": normalized}


def _validate_envar(value: Dynamic, ensemble_value: Dynamic) -> tuple[Config, Config]:
    algorithm = mapping(value, "algorithm")
    algorithm["prior_weight"] = nonnegative(
        algorithm.get("prior_weight", 1.0),
        "algorithm.prior_weight",
    )
    algorithm["outer_iterations"] = _positive_integer(
        algorithm.get("outer_iterations", 1),
        "algorithm.outer_iterations",
    )
    ensemble = mapping(ensemble_value, "ensemble")
    ensemble["workers"] = _positive_integer(
        ensemble.get("workers", 1),
        "ensemble.workers",
    )
    save_initial_states = ensemble.get("save_initial_states", False)
    if not isinstance(save_initial_states, bool):
        raise ValueError("ensemble.save_initial_states must be a boolean")
    ensemble["save_initial_states"] = save_initial_states
    modes = ensemble.get("modes")
    scales = ensemble.get("mode_scales")
    if modes is not None and scales is not None:
        raise ValueError("ensemble must specify either modes or mode_scales, not both")
    if modes is None:
        scales = mapping(scales, "ensemble.mode_scales")
        if not scales:
            raise ValueError("ensemble.mode_scales must not be empty")
        for name, mode_scale in scales.items():
            scales[name] = positive(mode_scale, f"ensemble.mode_scales.{name}")
        ensemble["mode_scales"] = scales
        modes = _legacy_modes(scales)
    ensemble["modes"] = _validate_ensemble_modes(modes)
    return algorithm, ensemble


def validate_inverse_case(
    case: Dynamic,
    base_dir: Optional[PathLike] = None,
) -> Config:
    """Validate and normalize a plain inverse-case mapping."""

    case = deepcopy(mapping(case, "case"))
    if case.get("format") != CASE_FORMAT:
        raise ValueError(f"case.format must be '{CASE_FORMAT}'")
    case = migrate_inverse_case(case, base_dir)
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
        case["algorithm"] = _validate_var(
            case.get("algorithm"),
            case["control"]["kind"],
        )
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
