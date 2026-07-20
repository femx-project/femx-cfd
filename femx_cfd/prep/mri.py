"""MRI loading and discrete-velocity projection onto an inlet."""

import h5py
import numpy as np

from femx_cfd._typing import Config, Dynamic, PathLike


def load_mri(path: PathLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        if "x" not in h5 or "u" not in h5 or "t" not in h5:
            raise ValueError("MRI HDF5 must contain x, u, and t")
        pts = np.asarray(h5["x"][...], dtype=float)
        vel_keys = sorted(h5["u"])
        time_keys = sorted(h5["t"])
        if vel_keys != time_keys or not vel_keys:
            raise ValueError("MRI velocity and time keys must match")
        times = np.asarray([h5["t"][key][()] for key in time_keys], dtype=float)
        vel = np.stack([h5["u"][key][...] for key in vel_keys])

    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise ValueError("MRI points must have shape (num_points, 3)")
    if vel.shape != (times.size, pts.shape[0], 3):
        raise ValueError("MRI velocity must have shape (num_times, num_points, 3)")
    if (
        not np.all(np.isfinite(pts))
        or not np.all(np.isfinite(times))
        or not np.all(np.isfinite(vel))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("MRI data must be finite with increasing times")
    return pts, times, np.asarray(vel, dtype=float)


def inlet_frame(
    surface: Dynamic,
    dim: int,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(surface.coordinates)[:, :dim]
    center = coords.mean(axis=0)
    centered = coords - center

    _, _, axes = np.linalg.svd(np.asarray(normal).reshape(1, dim))
    tangents = axes[1:].T
    local = centered @ tangents

    return center, tangents, local


def _line_stencils(
    local_nodes: np.ndarray,
    elements: np.ndarray,
    pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(local_nodes, dtype=float).reshape(-1)
    pts = np.asarray(pts, dtype=float).reshape(-1)
    stencils = np.zeros((pts.size, nodes.size))
    keep = np.ones(pts.size, dtype=bool)
    for row, pt in enumerate(pts):
        found = False
        for edge in np.asarray(elements, dtype=int):
            left, right = nodes[edge]
            length = right - left
            if abs(length) <= np.finfo(float).eps:
                continue
            weight = (pt - left) / length
            if -1.0e-8 <= weight <= 1.0 + 1.0e-8:
                stencils[row, edge] = (1.0 - weight, weight)
                found = True
                break
        keep[row] = found
    return stencils[keep], keep


def _triangle_stencils(
    local_nodes: np.ndarray,
    elements: np.ndarray,
    pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local_nodes = np.asarray(local_nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    pts = np.asarray(pts, dtype=float)
    stencils = np.zeros((pts.shape[0], local_nodes.shape[0]))
    keep = np.ones(pts.shape[0], dtype=bool)
    for row, pt in enumerate(pts):
        best = None
        best_margin = -np.inf
        for triangle in elements:
            a, b, c = local_nodes[triangle]
            e0 = b - a
            e1 = c - a
            rhs = pt - a
            det = e0[0] * e1[1] - e0[1] * e1[0]
            if abs(det) <= np.finfo(float).eps:
                continue
            r = (rhs[0] * e1[1] - rhs[1] * e1[0]) / det
            s = (e0[0] * rhs[1] - e0[1] * rhs[0]) / det
            wts = np.asarray((1.0 - r - s, r, s))
            margin = float(wts.min())
            if margin >= -1.0e-8 and margin > best_margin:
                best = (triangle, wts)
                best_margin = margin
        if best is None:
            keep[row] = False
            continue
        triangle, wts = best
        stencils[row, triangle] = wts
    return stencils[keep], keep


def _surface_stencils(
    local_nodes: np.ndarray,
    elements: np.ndarray,
    pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if np.asarray(elements).shape[1] == 2:
        return _line_stencils(local_nodes, elements, pts)
    return _triangle_stencils(local_nodes, elements, pts)


def project_to_inlet(
    surface: Dynamic,
    pts: np.ndarray,
    times: np.ndarray,
    vel: np.ndarray,
    normal: np.ndarray,
    opts: Config,
) -> np.ndarray:
    """Reconstruct all MRI velocity components on a smooth inlet field."""

    dim = pts.shape[1]
    center, tangents, local = inlet_frame(surface, dim, normal)
    rel = pts - center
    dist = rel @ normal
    proj = pts - dist[:, None] * normal
    proj_local = (proj - center) @ tangents
    selected = np.flatnonzero((dist >= -1.0e-8) & (dist <= opts["depth"]))
    stencils, keep = _surface_stencils(
        local,
        surface.elements,
        proj_local[selected],
    )
    selected = selected[keep]
    selected_dist = dist[selected]
    if selected.size < opts["laplacian_modes"] + 2:
        raise ValueError("too few inlet-near MRI points for the projection basis")

    eigvals, eigmodes = surface.laplacian_modes(opts["laplacian_modes"])
    poisson = surface.poisson_profile(normalize=False)
    basis = np.column_stack((poisson, eigmodes))
    scales = np.max(np.abs(basis), axis=0)
    scaled_basis = basis / scales
    sample_basis = stencils @ scaled_basis
    decay = opts["axial_decay"]
    wts = np.exp(-0.5 * np.square(selected_dist / decay))
    weighted_basis = np.sqrt(wts)[:, None] * sample_basis
    ridge = float(opts["ridge"])
    penalties = np.concatenate(([0.0], np.sqrt(ridge) * eigvals / eigvals[0]))
    lhs = np.vstack((weighted_basis, np.diag(penalties)))

    targets = vel[:, selected]
    rhs = np.asarray(
        [
            np.vstack(
                (
                    np.sqrt(wts)[:, None] * target,
                    np.zeros((basis.shape[1], dim)),
                )
            )
            for target in targets
        ]
    )
    temporal_weight = float(opts.get("temporal_weight", 0.0))
    if not np.isfinite(temporal_weight) or temporal_weight < 0.0:
        raise ValueError("projection.temporal_weight must be nonnegative")

    if temporal_weight == 0.0 or times.size == 1:
        coeff = np.asarray(
            [np.linalg.lstsq(lhs, target, rcond=None)[0] for target in rhs]
        )
    else:
        block_lhs = np.kron(np.eye(times.size), lhs)
        differences = np.zeros((times.size - 1, times.size))
        indices = np.arange(times.size - 1)
        steps = np.diff(times)
        differences[indices, indices] = -1.0 / steps
        differences[indices, indices + 1] = 1.0 / steps
        temporal_lhs = np.sqrt(temporal_weight) * np.kron(
            differences,
            np.eye(basis.shape[1]),
        )
        system = np.vstack((block_lhs, temporal_lhs))
        target = np.vstack(
            (
                rhs.reshape(-1, dim),
                np.zeros((temporal_lhs.shape[0], dim)),
            )
        )
        coeff = np.linalg.lstsq(system, target, rcond=None)[0].reshape(
            times.size,
            basis.shape[1],
            dim,
        )

    return np.einsum("nb,tbd->tnd", scaled_basis, coeff)
