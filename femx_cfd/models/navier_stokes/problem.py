"""Construction helpers for femx Navier--Stokes problems."""

import femx
import numpy as np

from femx_cfd._typing import Config, Dynamic


def add_bc(prob: Dynamic, cfg: Config) -> None:
    """Add fixed boundary conditions declared in a case configuration."""
    conditions = cfg["bc"]
    if not isinstance(conditions, list):
        raise ValueError("bc must be a list")

    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError("each boundary condition must be a mapping")
        try:
            boundary = condition["boundary"]
            field = condition["field"]
            space = condition["space"]
            time = condition["time"]
        except KeyError as error:
            raise ValueError(
                "each bc requires boundary, field, space, and time"
            ) from error
        if boundary == cfg["control"]["boundary"] and field == "velocity":
            continue
        if condition.get("components", "all") != "all":
            raise ValueError("component-selective fixed bc is not implemented")
        if space.get("kind") != "uniform" or time.get("kind") != "uniform":
            raise ValueError("fixed boundary conditions require uniform values")
        prob.add_bc(femx.DirichletBC(boundary, field, space["value"]))


def num_steps(cfg: Config) -> int:
    ratio = cfg["simulation"]["end_time"] / cfg["time"]["dt"]
    steps = int(round(ratio))
    if not np.isclose(ratio, steps, rtol=0.0, atol=1.0e-10):
        raise ValueError("simulation.end_time must be an integer multiple of time.dt")
    return steps


def make_problem(
    cfg: Config,
    *,
    times: np.ndarray,
    periodic: bool,
    normal: np.ndarray,
    init_state: Dynamic = None,
    init_mean: Dynamic = None,
    init_modes: Dynamic = None,
) -> Dynamic:
    """Build the Navier--Stokes model configured for an analysis case."""

    fluid = cfg["fluid"]
    model = femx.NavierStokesModel(
        cfg["mesh"],
        num_steps(cfg),
        cfg["time"]["dt"],
        rho=fluid["rho"],
        mu=fluid["mu"],
    )
    prob = femx.NavierStokesProblem(model)
    control = cfg["control"]
    add_bc(prob, cfg)

    if control["kind"] != "normal":
        raise ValueError("control.kind must be 'normal'")
    ctr = femx.NormalVelocityControl(
        control["boundary"],
        normal=normal,
        times=times,
        periodic=periodic,
    )
    prob.add_ctr(ctr)
    if init_state is not None:
        prob.set_initial_state(init_state)
    if init_modes is not None:
        prob.add_init_ctr(femx.InitialStateControl(init_modes, init_mean))
    prob.build()
    return prob
