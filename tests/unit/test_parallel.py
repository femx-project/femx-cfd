import os
import time
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from femx_cfd.runtime.parallel import Runtime, for_case
from femx_cfd.models.navier_stokes.ensemble import (
    _ensemble_workers,
    _parallel_map,
)


def square(value):
    time.sleep(0.05)
    return os.getpid(), value * value


class ParallelEnsembleTest(unittest.TestCase):
    def test_petsc_runtime_uses_world_collects_and_finalizes(self):
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "femx_cfd.runtime.parallel.femx.petsc_rank",
                    return_value=1,
                )
            )
            stack.enter_context(
                patch(
                    "femx_cfd.runtime.parallel.femx.petsc_size",
                    return_value=3,
                )
            )
            barrier = stack.enter_context(
                patch("femx_cfd.runtime.parallel.femx.petsc_barrier")
            )
            finalize = stack.enter_context(
                patch("femx_cfd.runtime.parallel.femx.petsc_finalize")
            )
            collect = stack.enter_context(patch("femx_cfd.runtime.parallel.gc.collect"))
            limiter = stack.enter_context(
                patch("femx_cfd.runtime.parallel.threadpool_limits")
            )
            runtime = for_case({"solver": {"backend": "petsc"}})
            runtime.barrier()
            runtime.finalize()

        self.assertEqual(runtime, Runtime("petsc", rank=1, size=3))
        self.assertFalse(runtime.is_root)
        barrier.assert_called_once_with()
        collect.assert_called_once_with()
        finalize.assert_called_once_with()
        limiter.assert_called_once_with(limits=1, user_api="blas")
        limiter.return_value.restore_original_limits.assert_called_once_with()

    def test_serial_petsc_does_not_limit_blas_threads(self):
        with (
            patch(
                "femx_cfd.runtime.parallel.femx.petsc_rank",
                return_value=0,
            ),
            patch(
                "femx_cfd.runtime.parallel.femx.petsc_size",
                return_value=1,
            ),
            patch("femx_cfd.runtime.parallel.threadpool_limits") as limiter,
        ):
            runtime = for_case({"solver": {"backend": "petsc"}})

        self.assertEqual(runtime, Runtime("petsc", rank=0, size=1))
        limiter.assert_not_called()

    def test_collect_does_not_finalize_reusable_petsc(self):
        with (
            patch("femx_cfd.runtime.parallel.gc.collect") as collect,
            patch("femx_cfd.runtime.parallel.femx.petsc_finalize") as finalize,
        ):
            Runtime("petsc").collect()

        collect.assert_called_once_with()
        finalize.assert_not_called()

    def test_mpi_petsc_uses_one_ensemble_worker(self):
        messages = []
        cfg = {
            "solver": {"backend": "petsc"},
            "ensemble": {"workers": 4},
        }

        with patch(
            "femx_cfd.models.navier_stokes.ensemble.femx.petsc_size",
            return_value=2,
        ):
            workers = _ensemble_workers(cfg, messages.append)

        self.assertEqual(workers, 1)
        self.assertIn("PETSc MPI", messages[0])

    def test_runs_members_in_two_processes(self):
        args = [(value,) for value in range(4)]
        names = [f"member {value}" for value in range(4)]

        results = _parallel_map(
            square,
            args,
            names,
            workers=2,
            report=lambda _message: None,
        )

        self.assertEqual([value for _, value in results], [0, 1, 4, 9])
        self.assertEqual(len({pid for pid, _ in results}), 2)


if __name__ == "__main__":
    unittest.main()
