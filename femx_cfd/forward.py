"""Stable forward-simulation case loading and execution."""

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
import tempfile

import femx
import gmsh
import numpy as np
from omegaconf import OmegaConf

from femx_cfd.models.navier_stokes.problem import make_problem
from femx_cfd.prep.inlet import analytic_inlet, inlet_normal, repeat_inlet


CASE_FORMAT = "femx-cfd-forward"
CASE_VERSION = 1
LENGTH_SCALES = {
    "mm": 1.0e-3,
    "cm": 1.0e-2,
    "m": 1.0,
}


@dataclass(frozen=True)
class ForwardSolution:
    """A completed forward trajectory and the model that owns its mesh."""

    model: object
    trajectory: object


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive number") from error
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive number")
    return number


def _multiple(value, unit, name):
    ratio = value / unit
    count = int(round(ratio))
    if count <= 0 or not np.isclose(ratio, count, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{name} must be an integer multiple of time.dt")
    return count


def _validate_inlet(inlet, dt, end_time):
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
        raise ValueError(
            "inlet.waveform.kind must be 'uniform' or 'harmonics'"
        )

    period = _positive(inlet.get("period", end_time), "inlet.period")
    _multiple(period, dt, "inlet.period")
    ratio = end_time / period
    cycles = int(round(ratio))
    if cycles <= 0 or not np.isclose(ratio, cycles, rtol=0.0, atol=1.0e-10):
        raise ValueError("time.end_time must be an integer multiple of inlet.period")

    inlet["normal"] = normal
    inlet["period"] = period
    inlet["waveform"] = waveform


def _validate_boundary_conditions(conditions, inlet_boundary):
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("boundary_conditions must be a nonempty list")

    names = set()
    for index, condition in enumerate(conditions):
        condition = _mapping(
            condition,
            f"boundary_conditions[{index}]",
        )
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
                raise ValueError(
                    f"boundary_conditions[{index}].value must be finite"
                )
            condition["value"] = value
            continue
        raise ValueError(
            f"boundary_conditions[{index}].kind must be 'no_slip' or 'pressure'"
        )


def validate_forward_case(case, base_dir=None):
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
    case["mesh"] = {
        "path": str(mesh),
        "length_unit": length_unit,
    }

    fluid = _mapping(case.get("fluid"), "fluid")
    fluid["rho"] = _positive(fluid.get("rho"), "fluid.rho")
    fluid["mu"] = _positive(fluid.get("mu"), "fluid.mu")

    time = _mapping(case.get("time"), "time")
    time["dt"] = _positive(time.get("dt"), "time.dt")
    time["end_time"] = _positive(time.get("end_time"), "time.end_time")
    _multiple(time["end_time"], time["dt"], "time.end_time")

    inlet = _mapping(case.get("inlet"), "inlet")
    _validate_inlet(inlet, time["dt"], time["end_time"])
    _validate_boundary_conditions(
        case.get("boundary_conditions"),
        inlet["boundary"],
    )

    solver = _mapping(case.get("solver", {}), "solver")
    backend = solver.get("backend", "petsc")
    if not isinstance(backend, str) or not backend:
        raise ValueError("solver.backend must be a nonempty string")
    petsc_options = solver.get("petsc_options")
    if petsc_options is not None and not isinstance(petsc_options, str):
        raise ValueError("solver.petsc_options must be a string")
    solver["backend"] = backend
    case["solver"] = solver
    return case


def load_forward_case(path):
    """Load a YAML case, resolving its mesh relative to the YAML file."""

    case_path = Path(path).expanduser().resolve()
    if not case_path.is_file():
        raise ValueError(f"case does not exist: {case_path}")
    loaded = OmegaConf.to_container(OmegaConf.load(case_path), resolve=True)
    return validate_forward_case(loaded, base_dir=case_path.parent)


def _fixed_conditions(case, dimension):
    conditions = []
    for condition in case["boundary_conditions"]:
        if condition["kind"] == "no_slip":
            field = "velocity"
            value = [0.0] * dimension
        else:
            field = "pressure"
            value = condition["value"]
        conditions.append(
            {
                "boundary": condition["boundary"],
                "field": field,
                "space": {"kind": "uniform", "value": value},
                "time": {"kind": "uniform"},
            }
        )
    return conditions


@contextmanager
def _mesh_in_meters(mesh_options):
    path = Path(mesh_options["path"])
    scale = LENGTH_SCALES[mesh_options["length_unit"]]
    if scale == 1.0:
        yield path
        return
    if gmsh.isInitialized():
        raise RuntimeError("mesh scaling requires ownership of the Gmsh runtime")

    with tempfile.TemporaryDirectory(prefix="femx-cfd-mesh-") as directory:
        scaled = Path(directory) / "mesh.msh"
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(str(path))
            gmsh.model.mesh.affineTransform(
                [
                    scale, 0.0, 0.0, 0.0,
                    0.0, scale, 0.0, 0.0,
                    0.0, 0.0, scale, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                ]
            )
            gmsh.write(str(scaled))
        finally:
            gmsh.finalize()
        yield scaled


def _make_forward_problem(case):
    with _mesh_in_meters(case["mesh"]) as mesh_path:
        mesh = femx.Mesh.read(mesh_path)
        inlet_options = case["inlet"]
        boundary = inlet_options["boundary"]
        surface = mesh.boundary(boundary)
        normal = inlet_normal(
            mesh,
            surface,
            inlet_options["normal"],
        )
        inlet = analytic_inlet(
            surface,
            {
                "T": inlet_options["period"],
                "dt": case["time"]["dt"],
            },
            normal,
            boundary,
            space=inlet_options["profile"],
            time=inlet_options["waveform"],
        )

        cycles = int(round(case["time"]["end_time"] / inlet_options["period"]))
        inlet = repeat_inlet(
            inlet,
            cycles,
            end_time=case["time"]["end_time"],
        )
        inlet = replace(
            inlet,
            times=inlet.times[1:],
            vals=inlet.vals[1:],
            periodic=False,
        )

        problem_case = {
            "mesh": str(mesh_path),
            "fluid": case["fluid"],
            "time": {"dt": case["time"]["dt"]},
            "simulation": {"end_time": case["time"]["end_time"]},
            "control": {"boundary": boundary, "kind": "normal"},
            "bc": _fixed_conditions(case, mesh.dimension),
        }
        problem = make_problem(
            problem_case,
            times=inlet.times,
            periodic=False,
            normal=inlet.normal,
        )
    return problem, inlet.params_for(problem)


def solve_forward(case, progress=None):
    """Solve one validated case and return its complete trajectory."""

    case = validate_forward_case(case)
    problem, parameters = _make_forward_problem(case)
    backend = case["solver"]["backend"]
    if backend not in femx.solver_backends():
        available = ", ".join(femx.solver_backends())
        raise ValueError(
            f"solver backend '{backend}' is unavailable; available: {available}"
        )
    solver = problem.create_solver(backend)
    if progress is None:
        trajectory = solver.solve(parameters)
    else:
        trajectory = solver.solve(parameters, progress=progress)
    return ForwardSolution(problem.model, trajectory)
