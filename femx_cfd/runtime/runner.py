"""Run portable CFD cases through one lifecycle and output contract."""

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from femx_cfd._typing import Config, Dynamic, PathLike, Progress
from femx_cfd.cases import (
    FORWARD_FORMAT,
    INVERSE_FORMAT,
    LoadedCase,
    load_case,
    snapshot_case,
)
from femx_cfd.cases.forward import inlet_boundary_condition
from femx_cfd.data import load_inlet, save_inlet
from femx_cfd.workflows.forward import stream_forward
from femx_cfd.workflows.inverse import solve_inverse

from . import output, parallel, progress


def result_path(
    case: Union[LoadedCase, PathLike],
    value: Optional[PathLike] = None,
) -> Path:
    """Resolve CLI, configured, and default result directories."""

    case_path = case.path if isinstance(case, LoadedCase) else Path(case)
    case_path = case_path.expanduser().resolve()
    if value is not None:
        return Path(value).expanduser().resolve()
    if isinstance(case, LoadedCase):
        configured = case.resolved["output"].get("directory")
        if configured is not None:
            return Path(configured)
    return case_path.parent / "result"


def _progress_sink(kind: str, enabled: bool) -> Progress:
    if not enabled:
        return progress.NullProgress()
    if kind == "json":
        return progress.JsonProgress()
    if kind == "console":
        return progress.ConsoleProgress()
    raise ValueError("progress must be 'console' or 'json'")


def _wall_shear_stress_statistics(case: Config) -> Dynamic:
    options = case["output"].get("wall_shear_stress")
    if options is None or "time_average" not in options:
        return None

    time_average = options["time_average"]
    end_time = float(case["time"]["end_time"])
    if time_average == "full_run":
        start_time = 0.0
    else:
        inlet_options = inlet_boundary_condition(case)
        if inlet_options["source"] == "analytic":
            period = float(inlet_options["period"])
        else:
            inlet = load_inlet(inlet_options["path"])
            if not inlet.periodic:
                raise ValueError(
                    "wall_shear_stress.time_average 'last_period' requires "
                    "a periodic inlet"
                )
            if not np.isclose(inlet.times[0], 0.0):
                raise ValueError("periodic inlet HDF5 times must start at zero")
            period = float(inlet.times[-1])
            if period <= 0.0:
                raise ValueError("periodic inlet HDF5 must contain a positive duration")
        cycles = int(round(end_time / period))
        if cycles <= 0 or not np.isclose(
            cycles * period,
            end_time,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ValueError(
                "time.end_time must be an integer multiple of the inlet period"
            )
        start_time = end_time - period
        if np.isclose(start_time, 0.0, rtol=0.0, atol=1.0e-10):
            start_time = 0.0

    return {
        "time_average": time_average,
        "start_time": start_time,
        "end_time": end_time,
    }


def _linear_system_collector(case: LoadedCase, destination: Path) -> Dynamic:
    if not case.resolved["output"].get("linear_systems"):
        return None
    try:
        from femx_ml.collect import LinearSystemCollector
    except ModuleNotFoundError as error:
        if error.name == "femx_ml":
            raise RuntimeError(
                "linear-system output requires the optional femx-ml package"
            ) from error
        raise

    solver = case.resolved["solver"]
    return LinearSystemCollector(
        destination / "linear_systems.h5",
        metadata={
            "case_format": case.format,
            "case_version": case.version,
            "solver_platform": solver["platform"],
            "solver_backend": solver["backend"],
        },
    )


def _run_forward(
    case: LoadedCase,
    destination: Path,
    prog: Progress,
    runtime: parallel.Runtime,
) -> None:
    prog.stage("forward: solving and streaming Navier-Stokes states")
    wall_shear_stress = case.resolved["output"].get("wall_shear_stress")
    requires_every_step = (
        wall_shear_stress is not None and "time_average" in wall_shear_stress
    )
    writer = (
        output.ForwardStreamWriter(
            destination,
            every=case.resolved["output"]["every"],
            wall_shear_stress=wall_shear_stress,
            wall_shear_stress_statistics=_wall_shear_stress_statistics(case.resolved),
        )
        if runtime.is_root
        else None
    )
    collector = _linear_system_collector(case, destination)
    try:
        stream_forward(
            case.resolved,
            every=1 if requires_every_step else case.resolved["output"]["every"],
            sample=None if writer is None else writer.sample,
            progress=prog,
            linear_system_observer=collector,
        )
        runtime.barrier()
        if collector is not None:
            collector.close()
        if writer is not None:
            prog.stage("output: finalizing flow.h5 and flow.xdmf")
            output.write_streamed_forward_result(
                destination,
                writer,
                linear_systems=collector is not None,
            )
    except Exception:
        if collector is not None:
            collector.abort()
        if writer is not None:
            writer.abort()
        raise


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
        (destination / "inlet").mkdir(parents=True, exist_ok=True)
        save_inlet(
            destination / "inlet" / "initial_velocity.h5",
            solution.initial_velocity,
        )
        output.write_inverse_result(
            destination,
            solution,
            every=case.resolved["output"]["every"],
            ensemble_basis=case.resolved["output"].get("ensemble_basis", "final"),
        )


def _run_loaded_case(
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
    output.require_available_result_directory(destination)
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
    output_path = result_path(case, destination)
    return _run_loaded_case(
        case,
        output_path,
        progress_kind=progress_kind,
        finalize_runtime=finalize_runtime,
    )


__all__ = ["result_path", "run_case"]
