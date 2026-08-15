"""Forward case format version 3 migration, validation, and path resolution."""

from copy import deepcopy
from typing import Optional

import numpy as np
from femx_cfd._typing import Config, Dynamic, PathLike
from femx_cfd.cases.migrations import migrate_forward_case
from femx_cfd.cases.validation import (
    integer_multiple,
    mapping,
    positive,
    reject_unknown,
    resolve_file,
    validate_fixed_boundary,
    validate_fluid,
    validate_mesh,
    validate_output,
    validate_profile,
    validate_solver,
    validate_waveform,
)


CASE_FORMAT = "femx-cfd-forward"
CASE_VERSION = 3


def _validate_output(
    value: Dynamic,
    base_dir: Optional[PathLike],
    *,
    num_steps: int,
    boundary_conditions: list[Config],
) -> Config:
    output = mapping({} if value is None else value, "output")
    reject_unknown(
        output,
        {"directory", "every", "linear_systems", "wall_shear_stress"},
        "output",
    )
    common = {
        name: option
        for name, option in output.items()
        if name in {"directory", "every"}
    }
    validated = validate_output(common, base_dir, num_steps=num_steps)

    linear_systems = output.get("linear_systems", False)
    if not isinstance(linear_systems, bool):
        raise ValueError("output.linear_systems must be a boolean")
    if linear_systems:
        validated["linear_systems"] = True

    if "wall_shear_stress" not in output:
        return validated
    options = mapping(output["wall_shear_stress"], "output.wall_shear_stress")
    reject_unknown(
        options,
        {"boundary", "time_average"},
        "output.wall_shear_stress",
    )
    boundary = options.get("boundary")
    if not isinstance(boundary, str) or not boundary:
        raise ValueError("output.wall_shear_stress.boundary must be a nonempty string")
    matches = [
        condition
        for condition in boundary_conditions
        if condition["boundary"] == boundary
    ]
    if (
        len(matches) != 1
        or matches[0].get("kind") != "velocity"
        or "value" not in matches[0]
        or np.any(np.asarray(matches[0]["value"], dtype=float) != 0.0)
    ):
        raise ValueError(
            "output.wall_shear_stress.boundary must name a fixed zero-velocity boundary"
        )
    wall_shear_stress = {"boundary": boundary}
    if "time_average" in options:
        time_average = options["time_average"]
        if time_average not in ("last_period", "full_run"):
            raise ValueError(
                "output.wall_shear_stress.time_average must be "
                "'last_period' or 'full_run'"
            )
        wall_shear_stress["time_average"] = time_average
    validated["wall_shear_stress"] = wall_shear_stress
    return validated


def _validate_inlet_condition(
    condition: Config,
    name: str,
    dt: float,
    end_time: float,
    base_dir: Optional[PathLike],
) -> None:
    boundary = condition.get("boundary")
    if not isinstance(boundary, str) or not boundary:
        raise ValueError(f"{name}.boundary must be a nonempty string")
    if condition.get("kind") != "velocity":
        raise ValueError(f"{name}.kind must be 'velocity' for an inlet source")

    source = condition.get("source")
    normal = condition.get("normal", "auto")
    if not isinstance(normal, str):
        try:
            normal = [float(value) for value in normal]
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name}.normal must be 'auto' or a vector") from error
    elif normal != "auto":
        raise ValueError(f"{name}.normal must be 'auto' or a vector")
    condition["normal"] = normal

    if source == "file":
        reject_unknown(
            condition,
            {"boundary", "kind", "source", "normal", "path"},
            name,
        )
        condition["path"] = resolve_file(
            condition.get("path"),
            f"{name}.path",
            base_dir,
        )
        return
    if source != "analytic":
        raise ValueError(f"{name}.source must be 'analytic' or 'file'")

    reject_unknown(
        condition,
        {
            "boundary",
            "kind",
            "source",
            "normal",
            "profile",
            "waveform",
            "period",
        },
        name,
    )
    condition["profile"] = validate_profile(
        condition.get("profile"),
        f"{name}.profile",
    )
    waveform = validate_waveform(
        condition.get("waveform", {"kind": "uniform", "value": 1.0}),
        f"{name}.waveform",
    )

    period = positive(condition.get("period", end_time), f"{name}.period")
    integer_multiple(period, dt, f"{name}.period")
    ratio = end_time / period
    cycles = int(round(ratio))
    if cycles <= 0 or not np.isclose(ratio, cycles, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"time.end_time must be an integer multiple of {name}.period")

    condition["period"] = period
    condition["waveform"] = waveform


def _validate_boundary_conditions(
    conditions: Dynamic,
    dt: float,
    end_time: float,
    base_dir: Optional[PathLike],
) -> list[Config]:
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("boundary_conditions must be a nonempty list")

    names: set[str] = set()
    inlet_count = 0
    for index, condition in enumerate(conditions):
        name = f"boundary_conditions[{index}]"
        condition = mapping(condition, name)
        boundary = condition.get("boundary")
        if not isinstance(boundary, str) or not boundary:
            raise ValueError(f"{name}.boundary must be a nonempty string")
        if boundary in names:
            raise ValueError(f"duplicate boundary condition: {boundary}")
        names.add(boundary)

        source = condition.get("source")
        if source not in ("analytic", "file"):
            raise ValueError(f"{name}.source must be 'analytic' or 'file'")

        if source == "file" or "profile" in condition:
            _validate_inlet_condition(condition, name, dt, end_time, base_dir)
            inlet_count += 1
        else:
            condition = validate_fixed_boundary(condition, name)
        conditions[index] = condition

    if inlet_count != 1:
        raise ValueError(
            "boundary_conditions must contain exactly one velocity condition "
            "with a profile or file source"
        )
    return conditions


def inlet_boundary_condition(case: Config) -> Config:
    """Return the validated source-driven inlet boundary condition."""

    matches = [
        condition
        for condition in case["boundary_conditions"]
        if condition.get("kind") == "velocity"
        and (condition.get("source") == "file" or "profile" in condition)
    ]
    if len(matches) != 1:
        raise ValueError(
            "boundary_conditions must contain exactly one velocity condition "
            "with a profile or file source"
        )
    return matches[0]


def validate_forward_case(
    case: Dynamic,
    base_dir: Optional[PathLike] = None,
) -> Config:
    """Validate and normalize a plain forward-case mapping."""

    case = deepcopy(mapping(case, "case"))
    if case.get("format") != CASE_FORMAT:
        raise ValueError(f"case.format must be '{CASE_FORMAT}'")
    case = migrate_forward_case(case, base_dir)
    if case.get("version") != CASE_VERSION:
        raise ValueError(f"case.version must be {CASE_VERSION}")
    reject_unknown(
        case,
        {
            "format",
            "version",
            "output",
            "mesh",
            "fluid",
            "time",
            "solver",
            "boundary_conditions",
        },
        "case",
    )

    case["mesh"] = validate_mesh(case.get("mesh"), base_dir)
    case["fluid"] = validate_fluid(case.get("fluid"))

    time = mapping(case.get("time"), "time")
    reject_unknown(time, {"dt", "end_time"}, "time")
    time["dt"] = positive(time.get("dt"), "time.dt")
    time["end_time"] = positive(time.get("end_time"), "time.end_time")
    steps = integer_multiple(time["end_time"], time["dt"], "time.end_time")

    case["boundary_conditions"] = _validate_boundary_conditions(
        case.get("boundary_conditions"),
        time["dt"],
        time["end_time"],
        base_dir,
    )

    case["solver"] = validate_solver(case.get("solver"))
    case["output"] = _validate_output(
        case.get("output"),
        base_dir,
        num_steps=steps,
        boundary_conditions=case["boundary_conditions"],
    )
    if case["output"].get("linear_systems") and (
        case["solver"]["platform"] != "cpu"
        or case["solver"]["backend"] not in ("dense", "resolve")
    ):
        raise ValueError(
            "output.linear_systems requires a CPU dense or ReSolve solver"
        )
    return case


__all__ = [
    "CASE_FORMAT",
    "CASE_VERSION",
    "inlet_boundary_condition",
    "validate_forward_case",
]
