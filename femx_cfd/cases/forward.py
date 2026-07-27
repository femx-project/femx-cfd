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
        raise ValueError(
            f"time.end_time must be an integer multiple of {name}.period"
        )

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
    case["output"] = validate_output(
        case.get("output"),
        base_dir,
        num_steps=steps,
    )
    return case


__all__ = [
    "CASE_FORMAT",
    "CASE_VERSION",
    "inlet_boundary_condition",
    "validate_forward_case",
]
