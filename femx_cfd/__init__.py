"""CFD forward, forecasting, and data-assimilation support for femx."""

from .data import (
    InletData,
    VelocityData,
    add_velocity_noise,
    load_inlet,
    load_velocity,
    make_velocity_obs,
    save_inlet,
    save_velocity,
)
from .cases import LoadedCase, load_case, validate_case
from .cases.forward import (
    CASE_FORMAT,
    CASE_VERSION,
    load_forward_case,
    validate_forward_case,
)
from .da import EnVar, EnVarResult, Var, VarResult
from .workflows.forward import ForwardSolution, solve_forward
from .workflows.inverse import InverseSolution, solve_inverse

__all__ = [
    "EnVar",
    "EnVarResult",
    "CASE_FORMAT",
    "CASE_VERSION",
    "ForwardSolution",
    "InletData",
    "InverseSolution",
    "LoadedCase",
    "VelocityData",
    "Var",
    "VarResult",
    "add_velocity_noise",
    "load_inlet",
    "load_forward_case",
    "load_case",
    "load_velocity",
    "make_velocity_obs",
    "save_inlet",
    "save_velocity",
    "solve_forward",
    "solve_inverse",
    "validate_case",
    "validate_forward_case",
]
