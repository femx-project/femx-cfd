"""Adapt femx execution-platform and solver-type selections for CFD workflows."""

import femx

from femx_cfd._typing import Dynamic


_PLATFORMS = {
    "cpu": femx.MemorySpace.HOST,
    "cuda": femx.MemorySpace.DEVICE,
}
_SOLVER_TYPES = {
    "dense": femx.SolverType.DENSE,
    "resolve": femx.SolverType.RESOLVE,
    "petsc": femx.SolverType.PETSC,
}


def _name(values: dict[str, Dynamic], value: Dynamic) -> str:
    return next(name for name, candidate in values.items() if candidate == value)


def _platform(value: Dynamic) -> Dynamic:
    if isinstance(value, str):
        try:
            return _PLATFORMS[value.lower()]
        except KeyError as error:
            raise ValueError("solver.platform must be 'cpu' or 'cuda'") from error
    if isinstance(value, femx.MemorySpace):
        return value
    raise TypeError("platform must be a string or MemorySpace")


def _solver_type(value: Dynamic) -> Dynamic:
    if isinstance(value, str):
        try:
            return _SOLVER_TYPES[value.lower()]
        except KeyError as error:
            raise ValueError(
                "solver.backend must be 'dense', 'resolve', or 'petsc'"
            ) from error
    if isinstance(value, femx.SolverType):
        return value
    raise TypeError("backend must be a string or SolverType")


def select_solver(
    platform: Dynamic,
    backend: Dynamic,
) -> tuple[Dynamic, Dynamic]:
    """Return a validated femx execution-platform and solver-type pair."""

    memory_space = _platform(platform)
    solver_type = _solver_type(backend)

    available = femx.solver_types(memory_space)
    if solver_type not in available:
        names = ", ".join(_name(_SOLVER_TYPES, value) for value in available)
        if not names:
            names = "none"
        raise ValueError(
            f"solver backend {_name(_SOLVER_TYPES, solver_type)!r} is unavailable "
            f"on {_name(_PLATFORMS, memory_space)!r} platform; available: {names}"
        )
    return memory_space, solver_type


def platform_name(value: Dynamic) -> str:
    """Return the configured execution-platform name."""

    return _name(_PLATFORMS, _platform(value))


def solver_name(value: Dynamic) -> str:
    """Return the configured solver-type name."""

    return _name(_SOLVER_TYPES, _solver_type(value))


__all__ = [
    "platform_name",
    "select_solver",
    "solver_name",
]
