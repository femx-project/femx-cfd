"""Truth solves and velocity observations for synthetic inverse cases."""

from dataclasses import replace

import femx
import numpy as np

from femx_cfd._typing import Config, Dynamic
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


def observation_points(case: Config) -> np.ndarray:
    options = case["observation"]
    x = np.linspace(options["lower"][0], options["upper"][0], options["counts"][0])
    y = np.linspace(options["lower"][1], options["upper"][1], options["counts"][1])
    return np.array([(px, py) for py in y for px in x])


def observation_times(case: Config, cycle: int = 1) -> np.ndarray:
    period = case["time"]["T"]
    count = case["observation"]["phases"]
    local = np.linspace(0.0, period, count, endpoint=False)
    return local + (cycle - 1) * period


def make_velocity_observations(case: Config, inlet: InletData) -> VelocityData:
    """Solve the truth case and sample its synthetic velocity observations."""

    problem, parameters, cycles = _make_truth_problem(case, inlet)
    truth = problem.solve(parameters, backend=case["solver"]["backend"])
    points = observation_points(case)
    times = observation_times(case)
    operator = femx.VelocityObservationOperator(
        problem.model,
        points,
        observation_times(case, cycle=cycles),
        components=range(problem.model.mesh.dimension),
    )
    return VelocityData(
        points=points,
        times=times,
        vals=operator.reshape(operator.predict(truth)),
    )
