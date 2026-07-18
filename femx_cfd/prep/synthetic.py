"""Synthetic truth and observation sampling for numerical verification."""

from dataclasses import replace

import numpy as np

from femx_cfd._typing import Config, Dynamic
from femx_cfd.data import InletData
from femx_cfd.models.navier_stokes.problem import make_problem

from .inlet import repeat_inlet


def make_truth_problem(
    cfg: Config,
    inlet: InletData,
) -> tuple[Dynamic, np.ndarray, int]:
    period = inlet.times[-1] - inlet.times[0]
    end_time = cfg["simulation"]["end_time"]
    cycles = int(round(end_time / period))
    inlet = repeat_inlet(inlet, cycles, end_time=end_time)
    # State solves begin at dt, so the non-periodic t=0 control level is not
    # sampled by any residual equation. Keep it in the prepared periodic input,
    # but omit it from the expanded truth problem.
    inlet = replace(inlet, times=inlet.times[1:], vals=inlet.vals[1:])
    prob = make_problem(
        cfg,
        times=inlet.times,
        periodic=False,
        normal=inlet.normal,
    )
    return prob, inlet.params_for(prob), cycles


def obs_points(cfg: Config) -> np.ndarray:
    obs = cfg["observation"]
    x = np.linspace(obs["lower"][0], obs["upper"][0], obs["counts"][0])
    y = np.linspace(obs["lower"][1], obs["upper"][1], obs["counts"][1])
    return np.array([(px, py) for py in y for px in x])


def obs_times(cfg: Config, cycle: int = 1) -> np.ndarray:
    T = cfg["time"]["T"]
    count = cfg["observation"]["phases"]
    local = np.linspace(0.0, T, count, endpoint=False)
    return local + (cycle - 1) * T
