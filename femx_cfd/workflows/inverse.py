"""Run Var and EnVar from one portable inverse case representation."""

from dataclasses import dataclass
from pathlib import Path

import femx
import numpy as np

from femx_cfd._typing import Config, Dynamic, Progress, ProgressCallback
from femx_cfd.cases.inverse import validate_inverse_case
from femx_cfd.da import EnVar, Var
from femx_cfd.data import (
    InletVelocityData,
    VelocityData,
    load_inlet,
    load_velocity,
    make_velocity_obs,
)
from femx_cfd.mesh import mesh_in_meters
from femx_cfd.models import navier_stokes as ns
from femx_cfd.models.navier_stokes.solver import select_solver
from femx_cfd.prep.inlet import (
    inlet_normal,
    normal_inlet,
    reconstruct_velocity_inlet,
)

from .boundaries import fixed_conditions


@dataclass(frozen=True)
class InverseSolution:
    """Artifacts and metrics produced by one inverse-analysis workflow."""

    model: Dynamic
    method: str
    values: Config
    trajectory: Dynamic
    metrics: Config
    initial_inlet: Dynamic
    initial_velocity: InletVelocityData
    predicted_velocity: VelocityData
    ensemble_bases: tuple[Dynamic, ...] = ()


def _predicted_velocity(velocity: VelocityData, values: Dynamic) -> VelocityData:
    """Restore an observation vector to its point and phase layout."""

    return VelocityData(
        points=velocity.points,
        times=velocity.times,
        vals=values.reshape(velocity.vals.shape),
    )


def _outer_progress(prog: Progress, label: str) -> ProgressCallback:
    """Prefix EnVar-generated stages with their outer-loop position."""

    def report(event: Dynamic) -> None:
        if event.get("type") == "stage":
            event = dict(event)
            message = event["message"].removeprefix("EnVar: ")
            event["message"] = f"{label}: {message}"
        prog(event)

    return report


def _problem_case(case: Config, mesh_path: Path, dimension: int) -> Config:
    problem = {
        "mesh": str(mesh_path),
        "fluid": case["fluid"],
        "time": {"T": case["time"]["period"], "dt": case["time"]["dt"]},
        "simulation": {"end_time": case["time"]["end_time"]},
        "control": case["control"],
        "bc": fixed_conditions(case, dimension),
        "solver": case["solver"],
        "output": case["output"],
    }
    if case["method"] == "var":
        problem["var"] = case["algorithm"]
    else:
        problem["envar"] = case["algorithm"]
        problem["ensemble"] = case["ensemble"]
    return problem


def _initial_inlet(
    case: Config,
    mesh: Dynamic,
) -> tuple[Dynamic, InletVelocityData]:
    options = case["initialization"]["inlet"]
    boundary = case["control"]["boundary"]
    surface = mesh.boundary(boundary)
    normal_value = (
        case["control"].get("normal", "auto")
        if case["control"]["kind"] == "normal"
        else "auto"
    )
    normal = inlet_normal(
        mesh,
        surface,
        normal_value,
    )
    if options["source"] == "file":
        velocity = load_inlet(options["path"])
    else:
        velocity = reconstruct_velocity_inlet(
            surface,
            {
                "file": case["observations"]["path"],
                "T": case["time"]["period"],
                "projection": options["projection"],
            },
            normal,
            boundary,
        )
    if velocity.boundary != boundary:
        raise ValueError("inlet HDF5 boundary does not match the configured boundary")
    if case["control"]["kind"] == "normal":
        return normal_inlet(velocity, normal, boundary), velocity
    return velocity, velocity


def _solve_var(
    case: Config,
    inlet: Dynamic,
    inlet_velocity: InletVelocityData,
    prog: Progress,
) -> InverseSolution:
    backend = case["solver"]["backend"]
    memory_space, solver_type = select_solver(
        case["solver"]["platform"],
        backend,
    )
    prob_options = {
        "times": inlet.times,
        "periodic": inlet.periodic,
        "normal": getattr(inlet, "normal", None),
    }

    prog.stage("spin-up: computing the initial velocity")
    spin_problem = ns.make_problem(case, **prob_options)
    background = inlet.params_for(spin_problem)
    spin = spin_problem.create_solver(
        memory_space,
        solver_type,
        options=case["solver"].get("options"),
    ).solve(background, progress=prog)

    prog.stage("input: loading prepared velocity observations")
    velocity = load_velocity(case["observations_path"])
    problem = ns.make_problem(
        case,
        init_state=spin.values[-1],
        **prob_options,
    )
    observation = make_velocity_obs(
        velocity,
        problem.model,
        time_weights=case.get("observation_time_weights"),
    )

    prog.stage("Var: optimizing the inlet velocity")
    options = case["var"]
    regularization = options["regularization"]
    penalty = ns.make_reg(
        problem,
        bg=background,
        bg_weight=regularization["background"],
        space_weight=regularization["space"],
        time_weight=regularization["time"],
    )
    analysis = Var(
        problem,
        observation,
        reg=penalty.obj,
        init_params=background,
        platform=memory_space,
        backend=solver_type,
        solver_options=case["solver"].get("options"),
    )
    bounds = [tuple(options["control_bounds"])] * problem.num_param
    result = analysis.solve(
        background,
        opt=options["optimizer"],
        bnds=bounds,
        max_itrs=options["max_iterations"],
        grad_tol=options["gradient_tolerance"],
        func_tol=options["function_tolerance"],
        prog=prog,
    )
    return InverseSolution(
        model=problem.model,
        method="var",
        values={
            "parameters": result.param,
            "predicted_observations": result.pred,
        },
        trajectory=result.traj,
        metrics={
            "platform": case["solver"]["platform"],
            "backend": backend,
            "optimizer": options["optimizer"],
            "objective": result.obj,
            "observation_misfit": result.obs_misfit,
            "regularization": result.reg,
            "gradient_norm": result.grad_norm,
            "iterations": result.num_iter,
            "converged": result.converged,
            "message": result.msg,
        },
        initial_inlet=inlet,
        initial_velocity=inlet_velocity,
        predicted_velocity=_predicted_velocity(velocity, result.pred),
    )


def _solve_envar(
    case: Config,
    inlet: Dynamic,
    inlet_velocity: InletVelocityData,
    prog: Progress,
) -> InverseSolution:
    options = case["envar"]
    prog.stage("input: loading prepared velocity observations")
    velocity = load_velocity(case["observations_path"])
    current_inlet = inlet
    outer_results = []
    ensemble_bases = []
    total = options["outer_iterations"]

    for outer in range(1, total + 1):
        label = f"EnVar outer loop {outer}/{total}"

        def stage(message: str) -> None:
            prog.stage(f"{label}: {message}")

        stage("building the initial-state basis")
        spin = ns.build_ensemble(
            case,
            current_inlet,
            stage,
            progress=prog,
        )
        problem = ns.make_problem(
            case,
            times=current_inlet.times,
            periodic=current_inlet.periodic,
            normal=getattr(current_inlet, "normal", None),
            init_mean=spin.init_mean,
            init_modes=spin.init_modes,
        )
        observation = make_velocity_obs(
            velocity,
            problem.model,
            time_weights=case.get("observation_time_weights"),
        )
        parameter_basis = ns.make_param_basis(
            problem,
            spin.ctr_mean,
            spin.ctr_modes,
        )
        stage("building the observation basis")
        observation_basis, trajectory_basis = ns.build_obs_basis(
            case,
            current_inlet,
            spin,
            velocity,
            parameter_basis,
            stage,
            progress=prog,
            capture_states=True,
        )
        ensemble_bases.append(trajectory_basis)

        stage("estimating ensemble coefficients")
        analysis = EnVar(
            problem,
            observation,
            parameter_basis,
            obs_basis=observation_basis,
            basis_forwards=parameter_basis.num_coefficients + 1,
            prior_weight=options["prior_weight"],
            platform=case["solver"]["platform"],
            backend=case["solver"]["backend"],
            solver_options=case["solver"].get("options"),
        )
        result = analysis.solve(prog=_outer_progress(prog, label))
        outer_results.append(result)

        if outer < total:
            begin = problem.ctr_param_offset
            end = begin + problem.num_ctr_param
            current_inlet = current_inlet.with_params_for(
                problem,
                result.param[begin:end],
            )

    result = outer_results[-1]
    return InverseSolution(
        model=problem.model,
        method="envar",
        values={
            "parameters": result.param,
            "coefficients": result.coeff,
            "parameter_history": np.stack([item.param for item in outer_results]),
            "coefficient_history": np.stack([item.coeff for item in outer_results]),
            "coefficient_names": np.asarray(trajectory_basis.coefficient_names),
            "coefficient_covariance": result.coefficient_covariance,
            "predicted_observations": result.pred,
            "linear_predicted_observations": result.linear_pred,
        },
        trajectory=result.traj,
        metrics={
            "platform": case["solver"]["platform"],
            "backend": case["solver"]["backend"],
            "num_modes": problem.num_init_param,
            "linear_objective": result.obj,
            "linear_observation_misfit": result.obs_misfit,
            "linear_prior_penalty": result.prior,
            "linear_gradient_norm": result.grad_norm,
            "nonlinear_observation_misfit": result.nonlinear_misfit,
            "outer_iterations": total,
            "linear_objective_history": np.asarray(
                [item.obj for item in outer_results]
            ),
            "linear_observation_misfit_history": np.asarray(
                [item.obs_misfit for item in outer_results]
            ),
            "linear_prior_penalty_history": np.asarray(
                [item.prior for item in outer_results]
            ),
            "linear_gradient_norm_history": np.asarray(
                [item.grad_norm for item in outer_results]
            ),
            "nonlinear_observation_misfit_history": np.asarray(
                [item.nonlinear_misfit for item in outer_results]
            ),
            "forward_solves": sum(item.num_forwards for item in outer_results),
        },
        initial_inlet=inlet,
        initial_velocity=inlet_velocity,
        predicted_velocity=_predicted_velocity(velocity, result.pred),
        ensemble_bases=tuple(ensemble_bases),
    )


def solve_inverse(case: Config, progress: Progress) -> InverseSolution:
    """Run the inverse method selected by one validated inverse case."""

    case = validate_inverse_case(case)
    select_solver(
        case["solver"]["platform"],
        case["solver"]["backend"],
    )

    with mesh_in_meters(case["mesh"]) as mesh_path:
        mesh = femx.Mesh.read(mesh_path)
        inlet, velocity = _initial_inlet(case, mesh)
        problem_case = _problem_case(case, mesh_path, mesh.dimension)
        problem_case["observations_path"] = case["observations"]["path"]
        problem_case["observation_time_weights"] = case["observations"].get(
            "time_weights"
        )
        if case["method"] == "var":
            return _solve_var(problem_case, inlet, velocity, progress)
        return _solve_envar(problem_case, inlet, velocity, progress)


__all__ = ["InverseSolution", "solve_inverse"]
