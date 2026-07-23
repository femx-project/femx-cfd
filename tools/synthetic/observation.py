"""Truth solves and velocity observations for synthetic inverse cases."""

from dataclasses import replace

import femx
import numpy as np

from femx_cfd._typing import Config, Dynamic, OptionalProgress
from femx_cfd.data import InletData, VelocityData
from femx_cfd.models.navier_stokes.problem import make_problem
from femx_cfd.prep.inlet import repeat_inlet


def _make_truth_problem(
    case: Config,
    inlet: InletData,
) -> tuple[Dynamic, np.ndarray, int]:
    period = inlet.times[-1] - inlet.times[0]
    end_time = case["simulation"]["end_time"]
    cycles = int(round(end_time / period))
    inlet = repeat_inlet(inlet, cycles, end_time=end_time)
    # State solves begin at dt, so the non-periodic t=0 control level is not
    # sampled by any residual equation. Keep it in the prepared periodic input,
    # but omit it from the expanded truth problem.
    inlet = replace(inlet, times=inlet.times[1:], vals=inlet.vals[1:])
    problem = make_problem(
        case,
        times=inlet.times,
        periodic=False,
        normal=inlet.normal,
    )
    return problem, inlet.params_for(problem), cycles


def _mesh_mask(points: np.ndarray, mesh: Dynamic) -> np.ndarray:
    """Retain lattice points contained in top-dimensional Gmsh elements."""

    try:
        import gmsh
    except ImportError as error:
        raise RuntimeError(
            "mesh-masked observations require Gmsh; install it with `pip install gmsh`"
        ) from error

    if not isinstance(mesh, (str, bytes)):
        raise ValueError("observation.mask 'mesh' requires a mesh path")

    owns_session = not gmsh.isInitialized()
    previous_model = ""
    mask_model = ""
    terminal = None
    try:
        if owns_session:
            gmsh.initialize()
        models = set(gmsh.model.list())
        if models:
            previous_model = gmsh.model.getCurrent()
        mask_model = "femx-cfd-observation-mask"
        while mask_model in models:
            mask_model += "-"
        gmsh.model.add(mask_model)
        gmsh.model.setCurrent(mask_model)
        terminal = gmsh.option.getNumber("General.Terminal")
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.merge(str(mesh))
        entities = gmsh.model.getEntities()
        if not entities:
            raise ValueError("observation mask mesh contains no elements")
        dimension = max(dim for dim, _ in entities)

        inside = np.zeros(points.shape[0], dtype=bool)
        for index, point in enumerate(points):
            coordinate = np.zeros(3)
            coordinate[: point.size] = point
            try:
                gmsh.model.mesh.getElementByCoordinates(
                    *coordinate,
                    dim=dimension,
                    strict=True,
                )
            except Exception as error:
                if "No element found" not in str(error):
                    raise
            else:
                inside[index] = True
        return points[inside]
    finally:
        if terminal is not None and gmsh.isInitialized():
            gmsh.option.setNumber("General.Terminal", terminal)
        if mask_model and gmsh.isInitialized() and mask_model in gmsh.model.list():
            gmsh.model.setCurrent(mask_model)
            gmsh.model.remove()
        if previous_model and gmsh.isInitialized():
            gmsh.model.setCurrent(previous_model)
        if owns_session and gmsh.isInitialized():
            gmsh.finalize()


def observation_points(case: Config) -> np.ndarray:
    """Build the Cartesian observation lattice and apply its optional mask."""

    options = case["observation"]
    x = np.linspace(options["lower"][0], options["upper"][0], options["counts"][0])
    y = np.linspace(options["lower"][1], options["upper"][1], options["counts"][1])
    points = np.array([(px, py) for py in y for px in x])
    mask = options.get("mask", "none")
    if mask == "none":
        return points
    if mask != "mesh":
        raise ValueError("observation.mask must be 'none' or 'mesh'")
    points = _mesh_mask(points, case.get("mesh"))
    if points.shape[0] == 0:
        raise ValueError("observation mask removed every candidate point")
    return points


def observation_times(case: Config, cycle: int = 1) -> np.ndarray:
    period = case["time"]["T"]
    count = case["observation"]["phases"]
    local = np.linspace(0.0, period, count, endpoint=False)
    selected = case["observation"].get(
        "selected_phases",
        range(1, count + 1),
    )
    return local[np.asarray(selected, dtype=int) - 1] + (cycle - 1) * period


def solve_truth_observations(
    case: Config,
    inlet: InletData,
    progress: OptionalProgress = None,
) -> tuple[Dynamic, Dynamic, VelocityData]:
    """Solve the complete truth trajectory and sample velocity observations."""
    problem, parameters, cycles = _make_truth_problem(case, inlet)
    solver = problem.create_solver(case["solver"]["backend"])
    truth = solver.solve(
        parameters,
        progress=progress,
    )
    points = observation_points(case)
    times = observation_times(case)
    operator = femx.VelocityObservationOperator(
        problem.model,
        points,
        observation_times(case, cycle=cycles),
        components=range(problem.model.mesh.dimension),
    )
    velocity = VelocityData(
        points=points,
        times=times,
        vals=operator.reshape(operator.predict(truth)),
    )
    return problem.model, truth, velocity
