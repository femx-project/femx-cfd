"""Initial-state and inlet ensembles for Navier--Stokes models."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import get_context
from queue import Empty

import femx
import numpy as np

from femx_cfd._typing import Config, Dynamic, OptionalProgress, ReportCallback
from femx_cfd.data import make_velocity_obs

from .problem import make_problem


@dataclass(frozen=True)
class Spinup:
    names: tuple[str, ...]
    init_mean: np.ndarray
    init_modes: np.ndarray
    ctr_mean: np.ndarray
    ctr_modes: np.ndarray


def _surface_mode(prob: Dynamic, index: int) -> np.ndarray:
    """Return a normalized Dirichlet surface mode on active control nodes."""
    surface = prob.model.mesh.boundary(prob.ctr.boundary)
    local = {int(node): idx for idx, node in enumerate(surface.mesh_node_ids)}
    control_nodes = np.asarray(prob.ctr_mesh_node_ids, dtype=np.int64)
    _, first = np.unique(control_nodes, return_index=True)
    control_nodes = control_nodes[np.sort(first)]
    try:
        active = np.asarray(
            [local[int(node)] for node in control_nodes],
            dtype=np.int64,
        )
    except KeyError as error:
        raise RuntimeError(
            "control node is not part of its boundary surface"
        ) from error
    _, modes = surface.laplacian_modes(index + 1)
    values = np.asarray(modes[active, index], dtype=float)
    scale = np.max(np.abs(values))
    if not np.isfinite(scale) or scale == 0.0:
        raise RuntimeError("inlet surface mode is degenerate")
    return values / scale


def _spatial_mode(prob: Dynamic, definition: Config, size: int) -> np.ndarray:
    """Evaluate one configured inlet-surface mode on control nodes."""
    kind = definition["kind"]
    if kind == "uniform":
        return np.ones(size)
    if kind == "surface_laplacian":
        return _surface_mode(prob, definition["index"])
    raise ValueError(f"unsupported ensemble spatial mode: {kind}")


def _temporal_mode(
    times: np.ndarray,
    period: float,
    definition: Config,
) -> np.ndarray:
    """Evaluate one configured periodic temporal mode."""
    kind = definition["kind"]
    if kind == "constant":
        return np.ones_like(times)
    angle = definition["order"] * 2.0 * np.pi * times / period
    angle += definition.get("phase", 0.0)
    if kind == "sine":
        return np.sin(angle)
    if kind == "cosine":
        return np.cos(angle)
    raise ValueError(f"unsupported ensemble temporal mode: {kind}")


def _analytic_ctr_modes(
    cfg: Config,
    prob: Dynamic,
    ctr_mean: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    times = np.asarray(prob.ctr.times)[: prob.num_ctr_levels]
    profile = ctr_mean.reshape(prob.param_shape)
    vector = prob.ctr.kind == "vector"
    dimension = int(prob.model.mesh.dimension)
    if vector:
        if profile.shape[1] % dimension:
            raise ValueError("vector control size must be divisible by mesh dimension")
        profile = np.linalg.norm(
            profile.reshape(profile.shape[0], -1, dimension),
            axis=2,
        )
    modes_cfg = cfg["ensemble"]["modes"]
    spatial_cfg = modes_cfg["spatial"]
    temporal_cfg = modes_cfg["temporal"]
    names = []
    modes = []
    for definition in modes_cfg["combined"]:
        name = definition["name"]
        spatial = _spatial_mode(
            prob,
            spatial_cfg[definition["spatial"]],
            profile.shape[1],
        )
        temporal = _temporal_mode(
            times,
            cfg["time"]["T"],
            temporal_cfg[definition["temporal"]],
        )
        vals = definition["scale"] * temporal[:, None] * profile * spatial
        if not vector:
            names.append(name)
            modes.append(vals.reshape(-1))
            continue
        for component, axis in enumerate("xyz"[:dimension]):
            component_vals = np.zeros((*vals.shape, dimension))
            component_vals[..., component] = vals
            names.append(f"{name}_{axis}")
            modes.append(component_vals.reshape(-1))
    return names, np.column_stack(modes)


def _init_modes_from_states(
    prob: Dynamic,
    mean: np.ndarray,
    states: list[np.ndarray],
) -> np.ndarray:
    modes = np.column_stack([state - mean for state in states])
    vel = np.zeros(prob.model.num_states, dtype=bool)
    vel[np.asarray(prob.model.velocity_dofs)] = True
    modes[~vel] = 0.0
    modes[prob.fixed_dofs] = 0.0
    modes[prob.ctr_state_dofs] = 0.0
    return np.ascontiguousarray(modes)


def _member_progress(
    progress: OptionalProgress,
    stage: str,
    member: str,
) -> OptionalProgress:
    if progress is None:
        return None

    def report(event: Dynamic) -> None:
        if event.get("type") != "solve":
            return
        progress(
            {
                "type": "ensemble_solve",
                "stage": stage,
                "member": member,
                "phase": event["phase"],
                "step": event["step"],
                "total": event["total"],
            }
        )

    return report


_WORKER_PROGRESS_QUEUE: Dynamic = None


def _set_worker_progress_queue(queue: Dynamic) -> None:
    global _WORKER_PROGRESS_QUEUE
    _WORKER_PROGRESS_QUEUE = queue


def _queue_member_progress(
    queue: Dynamic,
    stage: str,
    member: str,
    event: Dynamic,
) -> None:
    if event.get("type") != "solve":
        return
    queue.put(
        {
            "type": "ensemble_solve",
            "stage": stage,
            "member": member,
            "phase": event["phase"],
            "step": event["step"],
            "total": event["total"],
        }
    )


def _run_member(
    func: Dynamic,
    args: tuple[Dynamic, ...],
    stage: str,
    member: str,
) -> Dynamic:
    """Run one picklable ensemble task and relay solve events to its parent."""
    if _WORKER_PROGRESS_QUEUE is None:
        raise RuntimeError("ensemble worker progress queue is unavailable")
    return func(
        *args,
        progress=lambda event: _queue_member_progress(
            _WORKER_PROGRESS_QUEUE,
            stage,
            member,
            event,
        ),
    )


def _drain_progress(queue: Dynamic, progress: OptionalProgress) -> None:
    while True:
        try:
            progress(queue.get_nowait())
        except Empty:
            return


def _parallel_map(
    func: Dynamic,
    args: list[tuple[Dynamic, ...]],
    names: list[str],
    workers: int,
    report: ReportCallback,
    *,
    stage: str = "ensemble",
    progress: OptionalProgress = None,
) -> list[Dynamic]:
    if workers <= 0:
        raise ValueError("ensemble.workers must be positive")

    if workers == 1:
        results = []

        for idx, (arg, name) in enumerate(zip(args, names), start=1):
            results.append(func(*arg, progress=_member_progress(progress, stage, name)))
            report(f"ensemble: {name} complete ({idx}/{len(args)})")

        return results

    results = [None] * len(args)

    context = get_context("spawn")
    if progress is None:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {pool.submit(func, *arg): idx for idx, arg in enumerate(args)}
            completed = 0
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    idx = futures.pop(future)
                    results[idx] = future.result()
                    completed += 1
                    report(f"ensemble: {names[idx]} complete ({completed}/{len(args)})")
        return results

    queue = context.Queue()
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_set_worker_progress_queue,
            initargs=(queue,),
        ) as progress_pool:
            futures = {
                progress_pool.submit(_run_member, func, arg, stage, names[idx]): idx
                for idx, arg in enumerate(args)
            }
            completed = 0
            while futures:
                done, _ = wait(
                    futures,
                    timeout=0.05,
                    return_when=FIRST_COMPLETED,
                )
                _drain_progress(queue, progress)
                for future in done:
                    idx = futures.pop(future)
                    results[idx] = future.result()
                    completed += 1
                    report(f"ensemble: {names[idx]} complete ({completed}/{len(args)})")
            _drain_progress(queue, progress)
    finally:
        queue.close()

    return results


def _ensemble_workers(cfg: Config, report: ReportCallback) -> int:
    workers = int(cfg["ensemble"].get("workers", 1))
    if workers <= 0:
        raise ValueError("ensemble.workers must be positive")

    if cfg["solver"]["backend"] == "petsc" and femx.petsc_size() > 1:
        if workers != 1:
            report(
                "ensemble: using one member at a time because each "
                "forward solve uses PETSc MPI"
            )
        return 1
    return workers


def _spinup_state(
    cfg: Config,
    inlet: Dynamic,
    params: np.ndarray,
    progress: OptionalProgress = None,
) -> np.ndarray:
    prob = make_problem(
        cfg,
        times=inlet.times,
        periodic=inlet.periodic,
        normal=getattr(inlet, "normal", None),
    )
    solver = prob.create_solver(cfg["solver"]["backend"])
    return solver.solve(params, progress=progress).values[-1]


def build_ensemble(
    cfg: Config,
    inlet: Dynamic,
    report: ReportCallback = lambda _message: None,
    progress: OptionalProgress = None,
) -> Spinup:
    """Run spin-up members and build initial-state and inlet modes."""

    prob = make_problem(
        cfg,
        times=inlet.times,
        periodic=inlet.periodic,
        normal=getattr(inlet, "normal", None),
    )
    ctr_mean = inlet.params_for(prob)
    names, ctr_modes = _analytic_ctr_modes(cfg, prob, ctr_mean)
    params = [ctr_mean]
    params.extend(ctr_mean + ctr_modes[:, idx] for idx in range(len(names)))

    labels = ["spin-up mean", *(f"spin-up {name}" for name in names)]
    workers = _ensemble_workers(cfg, report)

    if workers == 1:
        solver = prob.create_solver(cfg["solver"]["backend"])
        states = []

        for idx, (label, param) in enumerate(zip(labels, params), start=1):
            report(f"{label} ({idx}/{len(params)})")
            states.append(
                solver.solve(
                    param,
                    progress=_member_progress(progress, "spin-up", label),
                ).values[-1]
            )
    else:
        args = [(cfg, inlet, param) for param in params]
        states = _parallel_map(
            _spinup_state,
            args,
            labels,
            workers,
            report,
            stage="spin-up",
            progress=progress,
        )

    init_mean = states[0].copy()
    return Spinup(
        names=tuple(names),
        init_mean=init_mean,
        init_modes=_init_modes_from_states(prob, init_mean, states[1:]),
        ctr_mean=ctr_mean,
        ctr_modes=ctr_modes,
    )


def _obs_prediction(
    cfg: Config,
    inlet: Dynamic,
    spin: Spinup,
    velocity: Dynamic,
    params: np.ndarray,
    progress: OptionalProgress = None,
) -> np.ndarray:
    prob = make_problem(
        cfg,
        times=inlet.times,
        periodic=inlet.periodic,
        normal=getattr(inlet, "normal", None),
        init_mean=spin.init_mean,
        init_modes=spin.init_modes,
    )
    obs = make_velocity_obs(velocity, prob.model)
    solver = prob.create_solver(cfg["solver"]["backend"])
    traj = solver.solve(params, progress=progress)
    return obs.op.predict(traj)


def build_obs_basis(
    cfg: Config,
    inlet: Dynamic,
    spin: Spinup,
    velocity: Dynamic,
    param_basis: Dynamic,
    report: ReportCallback = lambda _message: None,
    progress: OptionalProgress = None,
) -> Dynamic:
    """Build the EnVar observation basis with parallel forward solves."""

    mean = param_basis.mean
    modes = param_basis.perturbations
    params = [mean]
    params.extend(mean + modes[:, col] for col in range(modes.shape[1]))

    labels = ["observation mean"]
    labels.extend(f"observation member {idx + 1}" for idx in range(modes.shape[1]))
    args = [(cfg, inlet, spin, velocity, param) for param in params]
    predictions = _parallel_map(
        _obs_prediction,
        args,
        labels,
        _ensemble_workers(cfg, report),
        report,
        stage="observation",
        progress=progress,
    )

    mean_pred = predictions[0]
    obs_modes = np.column_stack([pred - mean_pred for pred in predictions[1:]])
    return femx.EnsembleBasis(mean_pred, obs_modes)


def make_param_basis(
    prob: Dynamic,
    ctr_mean: np.ndarray,
    ctr_modes: np.ndarray,
) -> Dynamic:
    """Combine initial-state and inlet modes into one affine basis."""

    count = ctr_modes.shape[1]
    if prob.num_init_param != count:
        raise ValueError("initial-state and inlet modes must use the same count")
    mean = np.concatenate((np.zeros(count), ctr_mean))
    modes = np.zeros((prob.num_param, count))
    modes[:count] = np.eye(count)
    modes[prob.ctr_param_offset :] = ctr_modes
    return femx.EnsembleBasis(mean, modes)
