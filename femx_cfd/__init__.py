"""CFD forward, forecasting, and data-assimilation support for femx."""

from .data import (
    InletData,
    InletVelocityData,
    VelocityData,
    add_velocity_noise,
    load_inlet,
    load_velocity,
    make_velocity_obs,
    save_inlet,
    save_velocity,
)
from .cases import LoadedCase, load_case
from .da import EnVar, EnVarResult, Var, VarResult
from .workflows.forward import ForwardSolution, solve_forward
from .workflows.inverse import InverseSolution, solve_inverse

__all__ = [
    "EnVar",
    "EnVarResult",
    "ForwardSolution",
    "InletData",
    "InletVelocityData",
    "InverseSolution",
    "LoadedCase",
    "VelocityData",
    "Var",
    "VarResult",
    "add_velocity_noise",
    "load_inlet",
    "load_case",
    "load_velocity",
    "make_velocity_obs",
    "save_inlet",
    "save_velocity",
    "solve_forward",
    "solve_inverse",
]
