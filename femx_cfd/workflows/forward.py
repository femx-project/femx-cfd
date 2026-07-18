"""Build and solve portable forward-simulation workflows."""

from dataclasses import dataclass, replace
import femx
import numpy as np

from femx_cfd._typing import Config, Dynamic, OptionalProgress
from femx_cfd.cases.forward import validate_forward_case
from femx_cfd.mesh import mesh_in_meters
from femx_cfd.models.navier_stokes.problem import make_problem
from femx_cfd.prep.inlet import analytic_inlet, inlet_normal, repeat_inlet

from .boundaries import fixed_conditions


@dataclass(frozen=True)
class ForwardSolution:
    """A completed forward trajectory and the model that owns its mesh."""

    model: Dynamic
    trajectory: Dynamic


def _make_forward_problem(case: Config) -> tuple[Dynamic, np.ndarray]:
    with mesh_in_meters(case["mesh"]) as mesh_path:
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
) -> ForwardSolution:
    """Solve one validated case and return its complete trajectory."""

    case = validate_forward_case(case)
    problem, parameters = _make_forward_problem(case)
    backend = case["solver"]["backend"]
    if backend not in femx.solver_backends():
        available = ", ".join(femx.solver_backends())
        raise ValueError(
            f"solver backend '{backend}' is unavailable; available: {available}"
        )
    solver = problem.create_solver(
        backend,
        options=case["solver"].get("options"),
    )
    if progress is None:
        trajectory = solver.solve(parameters)
    else:
        trajectory = solver.solve(parameters, progress=progress)
    return ForwardSolution(problem.model, trajectory)
