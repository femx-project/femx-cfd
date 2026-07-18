"""Initial-state and inlet ensembles for Navier--Stokes models."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context

import femx
import numpy as np

from femx_cfd._typing import Config, Dynamic, ReportCallback
from femx_cfd.data import make_velocity_obs

from .problem import make_problem


@dataclass(frozen=True)
class Spinup:
    init_mean: np.ndarray
    init_modes: np.ndarray
    ctr_mean: np.ndarray
    ctr_modes: np.ndarray


def _surface_mode(prob: Dynamic, index: int) -> np.ndarray:
    """Return a normalized Dirichlet surface mode on active control nodes."""
    surface = prob.model.mesh.boundary(prob.ctr.boundary)
    local = {int(node): idx for idx, node in enumerate(surface.mesh_node_ids)}
    try:
        active = np.asarray(
            [local[int(node)] for node in prob.ctr_mesh_node_ids],
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


def _analytic_ctr_modes(
    cfg: Config,
    prob: Dynamic,
    ctr_mean: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    times = np.asarray(prob.ctr.times)[: prob.num_ctr_levels]
    T = cfg["time"]["T"]
    omega = 2.0 * np.pi / T
    profile = ctr_mean.reshape(prob.param_shape)
    scales = cfg["ensemble"]["mode_scales"]
    needs_skew = any(name.startswith("skew_") for name in scales)
    needs_shape = any(name.startswith("shape_") for name in scales)
    skew = _surface_mode(prob, 1) if needs_skew else None
    shape = _surface_mode(prob, 0) if needs_shape else None
    uniform = np.ones(profile.shape[1])
    one = np.ones_like(times)
    sin1 = np.sin(omega * times)
    cos1 = np.cos(omega * times)
    sin2 = np.sin(2.0 * omega * times)
    cos2 = np.cos(2.0 * omega * times)
    defs = (
        ("flow_mean", one, uniform),
        ("flow_sin1", sin1, uniform),
        ("flow_cos1", cos1, uniform),
        ("flow_sin2", sin2, uniform),
        ("flow_cos2", cos2, uniform),
        ("skew_mean", one, skew),
        ("skew_sin1", sin1, skew),
        ("skew_cos1", cos1, skew),
        ("shape_mean", one, shape),
        ("shape_cos2", cos2, shape),
    )
    names = []
    modes = []
    for name, temporal, spatial in defs:
        if name not in scales:
            continue

        vals = scales[name] * temporal[:, None] * profile * spatial
        names.append(name)
        modes.append(vals.reshape(-1))
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


def _parallel_map(
    func: Dynamic,
    args: Dynamic,
    names: list[str],
    workers: int,
    report: ReportCallback,
) -> list[Dynamic]:
    if workers <= 0:
        raise ValueError("ensemble.workers must be positive")

    if workers == 1:
        results = []

        for idx, (arg, name) in enumerate(zip(args, names), start=1):
            results.append(func(*arg))
            report(f"ensemble: {name} complete ({idx}/{len(args)})")

        return results

    results = [None] * len(args)

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
    ) as pool:
        futures = {pool.submit(func, *arg): idx for idx, arg in enumerate(args)}

        for done, future in enumerate(as_completed(futures), start=1):
            idx = futures[future]
            results[idx] = future.result()
            report(f"ensemble: {names[idx]} complete ({done}/{len(args)})")

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
) -> np.ndarray:
    prob = make_problem(
        cfg,
        times=inlet.times,
        periodic=inlet.periodic,
        normal=inlet.normal,
    )
    solver = prob.create_solver(cfg["solver"]["backend"])
    return solver.solve(params).values[-1]


def build_ensemble(
    cfg: Config,
    inlet: Dynamic,
    report: ReportCallback = lambda _message: None,
) -> Spinup:
    """Run spin-up members and build initial-state and inlet modes."""

    prob = make_problem(
        cfg,
        times=inlet.times,
        periodic=inlet.periodic,
        normal=inlet.normal,
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
            states.append(solver.solve(param).values[-1])
    else:
        args = [(cfg, inlet, param) for param in params]
        states = _parallel_map(
            _spinup_state,
            args,
            labels,
            workers,
            report,
        )

    init_mean = states[0].copy()
    return Spinup(
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
) -> np.ndarray:
    prob = make_problem(
        cfg,
        times=inlet.times,
        periodic=inlet.periodic,
        normal=inlet.normal,
        init_mean=spin.init_mean,
        init_modes=spin.init_modes,
    )
    obs = make_velocity_obs(velocity, prob.model)
    solver = prob.create_solver(cfg["solver"]["backend"])
    traj = solver.solve(params)
    return obs.op.predict(traj)


def build_obs_basis(
    cfg: Config,
    inlet: Dynamic,
    spin: Spinup,
    velocity: Dynamic,
    param_basis: Dynamic,
    report: ReportCallback = lambda _message: None,
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
