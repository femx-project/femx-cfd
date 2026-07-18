"""Run Var and EnVar from one portable inverse case representation."""

from dataclasses import dataclass
from pathlib import Path

import femx

from femx_cfd._typing import Config, Dynamic, Progress
from femx_cfd.cases.inverse import validate_inverse_case
from femx_cfd.data import InletData, load_inlet, load_velocity, make_velocity_obs
from femx_cfd.envar import EnVar
from femx_cfd.mesh import mesh_in_meters
from femx_cfd.models import navier_stokes as ns
from femx_cfd.prep.inlet import inlet_normal, velocity_inlet
from femx_cfd.var import Var


@dataclass(frozen=True)
class InverseSolution:
    """Artifacts and metrics produced by one inverse-analysis workflow."""

    model: Dynamic
    method: str
    values: Config
    trajectory: Dynamic
    metrics: Config
    initial_inlet: InletData


def _fixed_conditions(case: Config, dimension: int) -> list[Config]:
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


def _problem_case(case: Config, mesh_path: Path, dimension: int) -> Config:
    problem = {
        "mesh": str(mesh_path),
        "fluid": case["fluid"],
        "time": {"T": case["time"]["period"], "dt": case["time"]["dt"]},
        "simulation": {"end_time": case["time"]["end_time"]},
        "control": case["control"],
        "bc": _fixed_conditions(case, dimension),
        "solver": case["solver"],
    }
    if case["method"] == "var":
        problem["var"] = case["algorithm"]
    else:
        problem["envar"] = case["algorithm"]
        problem["ensemble"] = case["ensemble"]
    return problem


def _initial_inlet(case: Config, mesh: Dynamic) -> InletData:
    options = case["initialization"]["inlet"]
    if options["source"] == "file":
        inlet = load_inlet(options["path"])
    else:
        boundary = case["control"]["boundary"]
        surface = mesh.boundary(boundary)
        normal = inlet_normal(
            mesh,
            surface,
            case["control"].get("normal", "auto"),
        )
        inlet = velocity_inlet(
            surface,
            {
                "file": case["observations"]["path"],
                "T": case["time"]["period"],
                "projection": options["projection"],
            },
            normal,
            boundary,
        )
    if inlet.boundary != case["control"]["boundary"]:
        raise ValueError("initial inlet boundary does not match control.boundary")
    return inlet


def _solve_var(case: Config, inlet: InletData, prog: Progress) -> InverseSolution:
    backend = case["solver"]["backend"]
    prob_options = {
        "times": inlet.times,
        "periodic": inlet.periodic,
        "normal": inlet.normal,
    }

    prog.stage("spin-up: computing the initial velocity")
    spin_problem = ns.make_problem(case, **prob_options)
    background = inlet.params_for(spin_problem)
    spin = spin_problem.create_solver(backend).solve(background, progress=prog)

    prog.stage("input: loading prepared velocity observations")
    velocity = load_velocity(case["observations_path"])
    problem = ns.make_problem(
        case,
        init_state=spin.values[-1],
        **prob_options,
    )
    observation = make_velocity_obs(velocity, problem.model)

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
        backend=backend,
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
        values={"param": result.param, "pred": result.pred},
        trajectory=result.traj,
        metrics={
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
    )


def _solve_envar(case: Config, inlet: InletData, prog: Progress) -> InverseSolution:
    options = case["envar"]

    prog.stage("ensemble: building the initial-state basis")
    spin = ns.build_ensemble(case, inlet, prog.stage)

    prog.stage("input: loading prepared velocity observations")
    velocity = load_velocity(case["observations_path"])
    problem = ns.make_problem(
        case,
        times=inlet.times,
        periodic=inlet.periodic,
        normal=inlet.normal,
        init_mean=spin.init_mean,
        init_modes=spin.init_modes,
    )
    observation = make_velocity_obs(velocity, problem.model)
    parameter_basis = ns.make_param_basis(
        problem,
        spin.ctr_mean,
        spin.ctr_modes,
    )

    prog.stage("ensemble: building the observation basis")
    observation_basis = ns.build_obs_basis(
        case,
        inlet,
        spin,
        velocity,
        parameter_basis,
        prog.stage,
    )

    prog.stage("EnVar: estimating ensemble coefficients")
    analysis = EnVar(
        problem,
        observation,
        parameter_basis,
        obs_basis=observation_basis,
        basis_forwards=parameter_basis.num_coefficients + 1,
        prior_weight=options["prior_weight"],
        backend=case["solver"]["backend"],
    )
    result = analysis.solve(prog=prog)
    return InverseSolution(
        model=problem.model,
        method="envar",
        values={
            "param": result.param,
            "coeff": result.coeff,
            "pred": result.pred,
            "linear_pred": result.linear_pred,
        },
        trajectory=result.traj,
        metrics={
            "backend": case["solver"]["backend"],
            "num_modes": problem.num_init_param,
            "linear_objective": result.obj,
            "linear_observation_misfit": result.obs_misfit,
            "linear_prior_penalty": result.prior,
            "linear_gradient_norm": result.grad_norm,
            "nonlinear_observation_misfit": result.nonlinear_misfit,
            "forward_solves": result.num_forwards,
        },
        initial_inlet=inlet,
    )


def solve_inverse(case: Config, progress: Progress) -> InverseSolution:
    """Run the inverse method selected by one validated inverse case."""

    case = validate_inverse_case(case)
    backend = case["solver"]["backend"]
    if backend not in femx.solver_backends():
        available = ", ".join(femx.solver_backends())
        raise ValueError(
            f"solver backend '{backend}' is unavailable; available: {available}"
        )

    with mesh_in_meters(case["mesh"]) as mesh_path:
        mesh = femx.Mesh.read(mesh_path)
        inlet = _initial_inlet(case, mesh)
        problem_case = _problem_case(case, mesh_path, mesh.dimension)
        problem_case["observations_path"] = case["observations"]["path"]
        if case["method"] == "var":
            return _solve_var(problem_case, inlet, progress)
        return _solve_envar(problem_case, inlet, progress)


__all__ = ["InverseSolution", "solve_inverse"]
