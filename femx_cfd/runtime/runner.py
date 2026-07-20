"""Run portable CFD cases through one lifecycle and output contract."""

import os
from pathlib import Path
from typing import Optional

from femx_cfd._typing import PathLike, Progress
from femx_cfd.cases import (
    FORWARD_FORMAT,
    INVERSE_FORMAT,
    LoadedCase,
    load_case,
    snapshot_case,
)
from femx_cfd.data import save_inlet
from femx_cfd.workflows.forward import solve_forward
from femx_cfd.workflows.inverse import solve_inverse

from . import output, parallel, progress


def result_path(case_path: PathLike, value: Optional[PathLike] = None) -> Path:
    """Resolve a result directory relative to its source case."""

    case_path = Path(case_path).expanduser().resolve()
    if value is None:
        return case_path.parent / "result"
    destination = Path(value).expanduser()
    if not destination.is_absolute():
        destination = case_path.parent / destination
    return destination.resolve()


def _progress_sink(kind: str, enabled: bool) -> Progress:
    if not enabled:
        return progress.NullProgress()
    if kind == "json":
        return progress.JsonProgress()
    if kind == "console":
        return progress.ConsoleProgress()
    raise ValueError("progress must be 'console' or 'json'")


def _run_forward(
    case: LoadedCase,
    destination: Path,
    prog: Progress,
    runtime: parallel.Runtime,
) -> None:
    prog.stage("forward: solving Navier-Stokes trajectory")
    solution = solve_forward(case.resolved, progress=prog)
    runtime.barrier()
    if runtime.is_root:
        prog.stage("output: writing flow.h5 and flow.xdmf")
        output.write_forward_result(destination, solution)


def _run_inverse(
    case: LoadedCase,
    destination: Path,
    prog: Progress,
    runtime: parallel.Runtime,
) -> None:
    solution = solve_inverse(case.resolved, prog)
    runtime.barrier()
    if runtime.is_root:
        prog.stage("output: writing inverse result and flow trajectory")
        save_inlet(destination / "initial_inlet.h5", solution.initial_velocity)
        output.write_inverse_result(destination, solution)


def run_loaded_case(
    case: LoadedCase,
    destination: PathLike,
    *,
    progress_kind: str = "console",
    finalize_runtime: bool = False,
) -> Path:
    """Execute one validated case and return its result directory.

    PETSc remains available for subsequent library calls unless
    ``finalize_runtime`` is enabled by a process-owning caller such as the CLI.
    """

    destination = Path(destination).expanduser().resolve()
    solver = case.resolved.get("solver", {})
    petsc_options = solver.get("petsc_options")
    if petsc_options:
        os.environ.setdefault("PETSC_OPTIONS", petsc_options)
    if case.format == INVERSE_FORMAT and case.resolved["method"] == "envar":
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "1")

    runtime = None
    output_owned = False
    try:
        runtime = parallel.for_case(case.resolved)
        if runtime.is_root:
            snapshot = snapshot_case(case, destination)
            status_format = (
                output.FORWARD_STATUS_FORMAT
                if case.format == FORWARD_FORMAT
                else output.INVERSE_STATUS_FORMAT
            )
            output.prepare_run(
                destination,
                snapshot,
                status_format=status_format,
            )
            output_owned = True

        with _progress_sink(progress_kind, runtime.is_root) as prog:
            if case.format == FORWARD_FORMAT:
                _run_forward(case, destination, prog, runtime)
            elif case.format == INVERSE_FORMAT:
                _run_inverse(case, destination, prog, runtime)
            else:
                raise ValueError(f"unsupported case format: {case.format!r}")
            runtime.barrier()
            if runtime.is_root:
                prog.stage(f"complete: {destination}")
        return destination
    except Exception as error:
        if (runtime is None or runtime.is_root) and output_owned:
            status_format = (
                output.FORWARD_STATUS_FORMAT
                if case.format == FORWARD_FORMAT
                else output.INVERSE_STATUS_FORMAT
            )
            output.write_status(
                destination,
                "failed",
                status_format=status_format,
                error=str(error),
            )
        raise
    finally:
        if runtime is not None:
            if finalize_runtime:
                runtime.finalize()
            else:
                runtime.collect()


def run_case(
    case_path: PathLike,
    destination: Optional[PathLike] = None,
    *,
    progress_kind: str = "console",
    finalize_runtime: bool = False,
) -> Path:
    """Load and run one portable CFD case without finalizing PETSc by default."""

    case = load_case(case_path)
    output_path = result_path(case.path, destination)
    return run_loaded_case(
        case,
        output_path,
        progress_kind=progress_kind,
        finalize_runtime=finalize_runtime,
    )


__all__ = ["result_path", "run_case", "run_loaded_case"]
