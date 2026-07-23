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
    validate_fixed_boundaries,
    validate_fluid,
    validate_mesh,
    validate_output,
    validate_profile,
    validate_solver,
    validate_waveform,
)


CASE_FORMAT = "femx-cfd-forward"
CASE_VERSION = 3


def _validate_inlet(
    inlet: Config,
    dt: float,
    end_time: float,
    base_dir: Optional[PathLike],
) -> None:
    boundary = inlet.get("boundary")
    if not isinstance(boundary, str) or not boundary:
        raise ValueError("inlet.boundary must be a nonempty string")

    source = inlet.get("source", "analytic")
    normal = inlet.get("normal", "auto")
    if not isinstance(normal, str):
        try:
            normal = [float(value) for value in normal]
        except (TypeError, ValueError) as error:
            raise ValueError("inlet.normal must be 'auto' or a vector") from error
    elif normal != "auto":
        raise ValueError("inlet.normal must be 'auto' or a vector")
    inlet["normal"] = normal

    if source == "file":
        reject_unknown(inlet, {"source", "boundary", "normal", "path"}, "inlet")
        inlet["source"] = source
        inlet["path"] = resolve_file(inlet.get("path"), "inlet.path", base_dir)
        return
    if source != "analytic":
        raise ValueError("inlet.source must be 'analytic' or 'file'")

    reject_unknown(
        inlet,
        {"source", "boundary", "normal", "profile", "waveform", "period"},
        "inlet",
    )
    inlet["profile"] = validate_profile(inlet.get("profile"), "inlet.profile")
    waveform = validate_waveform(
        inlet.get("waveform", {"kind": "uniform", "value": 1.0}),
        "inlet.waveform",
    )

    period = positive(inlet.get("period", end_time), "inlet.period")
    integer_multiple(period, dt, "inlet.period")
    ratio = end_time / period
    cycles = int(round(ratio))
    if cycles <= 0 or not np.isclose(ratio, cycles, rtol=0.0, atol=1.0e-10):
        raise ValueError("time.end_time must be an integer multiple of inlet.period")

    inlet["period"] = period
    inlet["source"] = source
    inlet["waveform"] = waveform


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
            "inlet",
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

    inlet = mapping(case.get("inlet"), "inlet")
    _validate_inlet(inlet, time["dt"], time["end_time"], base_dir)
    case["boundary_conditions"] = validate_fixed_boundaries(
        case.get("boundary_conditions"),
        excluded={inlet["boundary"]},
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
    "validate_forward_case",
]
