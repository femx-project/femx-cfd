"""Small PETSc/MPI helpers shared by CFD case workflows."""

import gc
from dataclasses import dataclass, field
from typing import Optional

import femx
from threadpoolctl import threadpool_limits

from femx_cfd._typing import Config


@dataclass(frozen=True)
class Runtime:
    """Collective execution context for a selected solver backend."""

    backend: Optional[str]
    rank: int = 0
    size: int = 1
    _blas_limiter: Optional[object] = field(
        default=None,
        compare=False,
        repr=False,
    )

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.backend == "petsc":
            femx.petsc_barrier()

    def collect(self) -> None:
        """Release deferred objects and restore the process BLAS limit."""

        if self.backend == "petsc":
            # pybind keep-alive relationships can defer PETSc object cleanup
            # until cyclic garbage collection even after an application run
            # has returned.
            gc.collect()
        self._restore_blas_threads()

    def finalize(self) -> None:
        if self.backend == "petsc":
            self.collect()
            femx.petsc_finalize()

    def _restore_blas_threads(self) -> None:
        if self._blas_limiter is None:
            return
        self._blas_limiter.restore_original_limits()
        object.__setattr__(self, "_blas_limiter", None)


def for_case(case: Config) -> Runtime:
    """Create the runtime after ``PETSC_OPTIONS`` has been configured."""

    backend = case.get("solver", {}).get("backend")
    if backend != "petsc":
        return Runtime(backend=backend)
    rank = femx.petsc_rank()
    size = femx.petsc_size()
    return Runtime(
        backend=backend,
        rank=rank,
        size=size,
        _blas_limiter=(
            threadpool_limits(limits=1, user_api="blas") if size > 1 else None
        ),
    )
