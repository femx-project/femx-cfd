"""Build and solve portable forward-simulation workflows."""

from dataclasses import dataclass, replace
import femx
import numpy as np

from femx_cfd._typing import Config, Dynamic, OptionalProgress
from femx_cfd.cases.forward import (
    inlet_boundary_condition,
    validate_forward_case,
)
from femx_cfd.data import InletData, load_inlet
from femx_cfd.mesh import mesh_in_meters
from femx_cfd.models.navier_stokes.problem import make_problem
from femx_cfd.models.navier_stokes.solver import select_solver
from femx_cfd.prep.inlet import (
    analytic_inlet,
    inlet_normal,
    normal_inlet,
    repeat_inlet,
)

from .boundaries import fixed_conditions


@dataclass(frozen=True)
class ForwardSolution:
    """A completed forward trajectory and the model that owns its mesh."""

    model: Dynamic
    trajectory: Dynamic


def _file_inlet(case: Config, normal: np.ndarray) -> InletData:
    """Load and normalize a prepared inlet for a forward time interval."""

    options = inlet_boundary_condition(case)
    inlet = normal_inlet(
        load_inlet(options["path"]),
        normal,
        options["boundary"],
    )

    dt = case["time"]["dt"]
    end_time = case["time"]["end_time"]
    steps = int(round(end_time / dt))
    expected_times = np.linspace(dt, end_time, steps)

    if inlet.periodic:
        if not np.isclose(inlet.times[0], 0.0):
            raise ValueError("periodic inlet HDF5 times must start at zero")
        if not np.allclose(np.diff(inlet.times), dt, rtol=0.0, atol=1.0e-10):
            raise ValueError("inlet HDF5 times must use time.dt")
        period = inlet.times[-1]
        if period <= 0.0:
            raise ValueError("periodic inlet HDF5 must contain a positive duration")
        cycles = int(round(end_time / period))
        if cycles <= 0 or not np.isclose(
            cycles * period,
            end_time,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ValueError(
                "time.end_time must be an integer multiple of the "
                "periodic inlet HDF5 duration"
            )
        inlet = repeat_inlet(inlet, cycles, end_time=end_time)
    if np.isclose(inlet.times[0], 0.0):
        inlet = replace(
            inlet,
            times=inlet.times[1:],
            vals=inlet.vals[1:],
        )

    if inlet.times.shape != expected_times.shape or not np.allclose(
        inlet.times,
        expected_times,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError(
            "nonperiodic inlet HDF5 times must cover time.dt through "
            "time.end_time at time.dt intervals"
        )
    return replace(inlet, periodic=False)


def _make_forward_problem(case: Config) -> tuple[Dynamic, np.ndarray]:
    with mesh_in_meters(case["mesh"]) as mesh_path:
        mesh = femx.Mesh.read(mesh_path)
        inlet_options = inlet_boundary_condition(case)
        boundary = inlet_options["boundary"]
        surface = mesh.boundary(boundary)
        normal = inlet_normal(
            mesh,
            surface,
            inlet_options["normal"],
        )
        if inlet_options["source"] == "file":
            inlet = _file_inlet(case, normal)
        else:
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
            "bc": fixed_conditions(case, mesh.dimension),
        }
        problem = make_problem(
            problem_case,
            times=inlet.times,
            periodic=False,
            normal=inlet.normal,
        )
    return problem, inlet.params_for(problem)


def solve_forward(
    case: Config,
    progress: OptionalProgress = None,
    linear_system_observer: Dynamic = None,
) -> ForwardSolution:
    """Solve one validated case and return its complete trajectory."""

    case = validate_forward_case(case)
    problem, parameters = _make_forward_problem(case)
    memory_space, solver_type = select_solver(
        case["solver"]["platform"],
        case["solver"]["backend"],
    )
    solver_options = {
        "options": case["solver"].get("options"),
    }
    if linear_system_observer is not None:
        solver_options["linear_system_observer"] = linear_system_observer
    solver = problem.create_solver(memory_space, solver_type, **solver_options)
    if progress is None:
        trajectory = solver.solve(parameters)
    else:
        trajectory = solver.solve(parameters, progress=progress)
    return ForwardSolution(problem.model, trajectory)


def stream_forward(
    case: Config,
    every: int,
    sample: Dynamic = None,
    progress: OptionalProgress = None,
    linear_system_observer: Dynamic = None,
) -> Dynamic:
    """Solve one case without retaining its complete trajectory."""

    case = validate_forward_case(case)
    problem, parameters = _make_forward_problem(case)
    memory_space, solver_type = select_solver(
        case["solver"]["platform"],
        case["solver"]["backend"],
    )
    solver_options = {
        "options": case["solver"].get("options"),
    }
    if linear_system_observer is not None:
        solver_options["linear_system_observer"] = linear_system_observer
    solver = problem.create_solver(memory_space, solver_type, **solver_options)
    callback = (
        None
        if sample is None
        else lambda level, state: sample(problem.model, level, state)
    )
    solver.run(
        parameters,
        sample_every=every,
        sample=callback,
        progress=progress,
    )
    return problem.model
