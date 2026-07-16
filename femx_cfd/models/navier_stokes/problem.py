"""Construction helpers for femx Navier--Stokes problems."""

import femx
import numpy as np


def add_bc(prob, cfg):
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


def num_steps(cfg):
    ratio = cfg["simulation"]["end_time"] / cfg["time"]["dt"]
    steps = int(round(ratio))
    if not np.isclose(ratio, steps, rtol=0.0, atol=1.0e-10):
        raise ValueError("simulation.end_time must be an integer multiple of time.dt")
    return steps


def make_problem(
    cfg,
    *,
    times,
    periodic,
    normal,
    init_state=None,
    init_mean=None,
    init_modes=None,
):
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

    if control["kind"] == "normal":
        ctr = femx.NormalVelocityControl(
            control["boundary"],
            normal=normal,
            times=times,
            periodic=periodic,
        )
    elif control["kind"] == "vector":
        ctr = femx.VelocityControl(
            control["boundary"],
            times=times,
            periodic=periodic,
        )
    else:
        raise ValueError("control.kind must be 'normal' or 'vector'")
    prob.add_ctr(ctr)
    if init_state is not None:
        prob.set_initial_state(init_state)
    if init_modes is not None:
        prob.add_init_ctr(femx.InitialStateControl(init_modes, init_mean))
    prob.build()
    return prob
