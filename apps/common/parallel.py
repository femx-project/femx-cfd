"""Small PETSc/MPI runtime helpers for the application entry points."""

import gc
from dataclasses import dataclass
from typing import Optional

import femx


@dataclass(frozen=True)
class Runtime:
    """Collective execution context for a selected solver backend."""

    backend: Optional[str]
    rank: int = 0
    size: int = 1

    @property
    def is_root(self):
        return self.rank == 0

    @property
    def distributed(self):
        return self.size > 1

    def barrier(self):
        if self.backend == "petsc":
            femx.petsc_barrier()

    def finalize(self):
        if self.backend == "petsc":
            # pybind keep-alive relationships can defer PETSc object cleanup
            # until cyclic garbage collection even after an application run
            # has returned.
            gc.collect()
            femx.petsc_finalize()


def for_case(case):
    """Create the runtime after ``PETSC_OPTIONS`` has been configured."""

    backend = case.get("solver", {}).get("backend")
    if backend != "petsc":
        return Runtime(backend=backend)
    return Runtime(
        backend=backend,
        rank=femx.petsc_rank(),
        size=femx.petsc_size(),
    )
