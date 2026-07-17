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
from .envar import EnVar, EnVarResult
from .forward import (
    CASE_FORMAT,
    CASE_VERSION,
    ForwardSolution,
    load_forward_case,
    solve_forward,
    validate_forward_case,
)
from .var import Var, VarResult

__all__ = [
    "EnVar",
    "EnVarResult",
    "CASE_FORMAT",
    "CASE_VERSION",
    "ForwardSolution",
    "InletData",
    "VelocityData",
    "Var",
    "VarResult",
    "add_velocity_noise",
    "load_inlet",
    "load_forward_case",
    "load_velocity",
    "make_velocity_obs",
    "save_inlet",
    "save_velocity",
    "solve_forward",
    "validate_forward_case",
]
