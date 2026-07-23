"""Inverse-analysis case validation, migration, and path resolution."""

from copy import deepcopy
import math
from typing import Optional

from femx_cfd._typing import Config, Dynamic, PathLike
from femx_cfd.cases.migrations import migrate_inverse_case

from .validation import (
    integer_multiple,
    mapping,
    nonnegative,
    positive,
    positive_integer,
    reject_unknown,
    resolve_file,
    validate_fixed_boundaries,
    validate_fluid,
    validate_mesh,
    validate_output,
    validate_solver,
)


CASE_FORMAT = "femx-cfd-inverse"
CASE_VERSION = 3
_METHODS = {"var", "envar"}


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
    reject_unknown(control, {"boundary", "kind", "normal"}, "control")
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
    reject_unknown(time, {"period", "dt", "end_time"}, "time")
    period = positive(time.get("period"), "time.period")
    dt = positive(time.get("dt"), "time.dt")
    end_time = positive(time.get("end_time", period), "time.end_time")
    integer_multiple(period, dt, "time.period")
    integer_multiple(end_time, dt, "time.end_time")
    time.update({"period": period, "dt": dt, "end_time": end_time})
    return time


def _validate_output(
    value: Dynamic,
    base_dir: Optional[PathLike],
    *,
    num_steps: int,
    method: str,
) -> Config:
    output = mapping({} if value is None else value, "output")
    allowed = {"directory", "every"}
    if method == "envar":
        allowed.add("ensemble_basis")
    reject_unknown(output, allowed, "output")

    common = {
        name: val for name, val in output.items() if name in {"directory", "every"}
    }
    validated = validate_output(common, base_dir, num_steps=num_steps)
    if method == "envar":
        ensemble_basis = output.get("ensemble_basis", "final")
        if ensemble_basis not in ("final", "all"):
            raise ValueError("output.ensemble_basis must be 'final' or 'all'")
        validated["ensemble_basis"] = ensemble_basis
    return validated


def _validate_observations(value: Dynamic, base_dir: Optional[PathLike]) -> Config:
    observations = mapping(value, "observations")
    reject_unknown(
        observations,
        {"kind", "path", "time_weights"},
        "observations",
    )
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
            raise ValueError("observations.time_weights must be a nonempty list")
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
    reject_unknown(initialization, {"inlet"}, "initialization")
    inlet = mapping(initialization.get("inlet"), "initialization.inlet")
    source = inlet.get("source", "file")
    if source == "file":
        reject_unknown(
            inlet,
            {"source", "path"},
            "initialization.inlet",
        )
        inlet["path"] = resolve_file(
            inlet.get("path"),
            "initialization.inlet.path",
            base_dir,
        )
    elif source == "observations":
        reject_unknown(
            inlet,
            {"source", "projection"},
            "initialization.inlet",
        )
        projection = mapping(
            inlet.get("projection"),
            "initialization.inlet.projection",
        )
        reject_unknown(
            projection,
            {
                "depth",
                "axial_decay",
                "laplacian_modes",
                "ridge",
                "temporal_weight",
            },
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
        projection["laplacian_modes"] = positive_integer(
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
    reject_unknown(
        algorithm,
        {
            "optimizer",
            "max_iterations",
            "gradient_tolerance",
            "function_tolerance",
            "control_bounds",
            "regularization",
        },
        "algorithm",
    )
    optimizer = algorithm.get("optimizer", "tao")
    if optimizer not in ("scipy", "tao"):
        raise ValueError("algorithm.optimizer must be 'scipy' or 'tao'")
    algorithm["optimizer"] = optimizer
    algorithm["max_iterations"] = positive_integer(
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
    reject_unknown(
        regularization,
        {"background", "space", "time"},
        "algorithm.regularization",
    )
    for name in ("background", "space", "time"):
        regularization[name] = nonnegative(
            regularization.get(name, 0.0),
            f"algorithm.regularization.{name}",
        )
    algorithm["regularization"] = regularization
    return algorithm


def _validate_ensemble_modes(value: Dynamic) -> Config:
    modes = mapping(value, "ensemble.modes")
    reject_unknown(
        modes,
        {"spatial", "temporal", "combined"},
        "ensemble.modes",
    )
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
            reject_unknown(
                definition,
                {"kind"},
                f"ensemble.modes.spatial.{name}",
            )
        elif kind == "surface_laplacian":
            reject_unknown(
                definition,
                {"kind", "index"},
                f"ensemble.modes.spatial.{name}",
            )
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
            reject_unknown(
                definition,
                {"kind"},
                f"ensemble.modes.temporal.{name}",
            )
        elif kind in ("sine", "cosine"):
            reject_unknown(
                definition,
                {"kind", "order", "phase"},
                f"ensemble.modes.temporal.{name}",
            )
            definition["order"] = positive_integer(
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
        reject_unknown(
            definition,
            {"name", "spatial", "temporal", "scale"},
            prefix,
        )
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
    reject_unknown(
        algorithm,
        {"prior_weight", "outer_iterations"},
        "algorithm",
    )
    algorithm["prior_weight"] = nonnegative(
        algorithm.get("prior_weight", 1.0),
        "algorithm.prior_weight",
    )
    algorithm["outer_iterations"] = positive_integer(
        algorithm.get("outer_iterations", 1),
        "algorithm.outer_iterations",
    )
    ensemble = mapping(ensemble_value, "ensemble")
    reject_unknown(
        ensemble,
        {"workers", "profile_scaling", "modes"},
        "ensemble",
    )
    ensemble["workers"] = positive_integer(
        ensemble.get("workers", 1),
        "ensemble.workers",
    )
    profile_scaling = ensemble.get("profile_scaling", "instantaneous")
    if profile_scaling not in ("instantaneous", "peak"):
        raise ValueError("ensemble.profile_scaling must be 'instantaneous' or 'peak'")
    ensemble["profile_scaling"] = profile_scaling
    ensemble["modes"] = _validate_ensemble_modes(ensemble.get("modes"))
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
    reject_unknown(
        case,
        {
            "format",
            "version",
            "method",
            "output",
            "mesh",
            "fluid",
            "time",
            "solver",
            "control",
            "boundary_conditions",
            "observations",
            "initialization",
            "algorithm",
            "ensemble",
        },
        "case",
    )
    method = case.get("method")
    if method not in _METHODS:
        choices = ", ".join(sorted(_METHODS))
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
    steps = integer_multiple(
        case["time"]["end_time"],
        case["time"]["dt"],
        "time.end_time",
    )
    case["output"] = _validate_output(
        case.get("output"),
        base_dir,
        num_steps=steps,
        method=method,
    )
    if method == "var":
        if "ensemble" in case:
            raise ValueError("case.ensemble is only valid for method: envar")
        case["algorithm"] = _validate_var(
            case.get("algorithm"),
            case["control"]["kind"],
        )
    else:
        case["algorithm"], case["ensemble"] = _validate_envar(
            case.get("algorithm"),
            case.get("ensemble"),
        )
    return case


__all__ = [
    "CASE_FORMAT",
    "CASE_VERSION",
    "validate_inverse_case",
]
