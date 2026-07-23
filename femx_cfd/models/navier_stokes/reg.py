"""Navier--Stokes regularization for Var."""

from dataclasses import dataclass

import femx
import numpy as np
from scipy import sparse

from femx_cfd._typing import Dynamic


def _weight(val: float, name: str) -> float:
    val = float(val)
    if not np.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return val


def _params(vals: Dynamic, size: int, name: str) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    if arr.size != size:
        raise ValueError(f"{name} must contain {size} values")
    arr = np.ascontiguousarray(arr).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr.copy()


def _block_reg(
    prob: Dynamic,
    mat: Dynamic,
    weight: float,
    ref: np.ndarray,
) -> Dynamic:
    coo = mat.tocoo()
    levels, size = prob.param_shape
    return femx.TimeBlockRegularization(
        prob.model.num_steps,
        prob.model.num_states,
        levels,
        size,
        np.asarray(coo.row, dtype=np.int64),
        np.asarray(coo.col, dtype=np.int64),
        np.asarray(coo.data, dtype=float),
        weight,
        ref,
    )


def _full_reg(
    prob: Dynamic,
    mat: Dynamic,
    weight: float,
    ref: np.ndarray,
) -> Dynamic:
    coo = mat.tocoo()
    return femx.TimeBlockRegularization(
        prob.model.num_steps,
        prob.model.num_states,
        1,
        prob.num_param,
        np.asarray(coo.row, dtype=np.int64),
        np.asarray(coo.col, dtype=np.int64),
        np.asarray(coo.data, dtype=float),
        weight,
        ref,
    )


def _combined_mats(
    prob: Dynamic,
    stiffness: Dynamic,
    mass: Dynamic,
) -> tuple[Dynamic, Dynamic, Dynamic, Dynamic]:
    levels, size = prob.param_shape
    num_init = prob.num_init_param
    init = sparse.eye(num_init, format="csr")
    zeros = sparse.csr_matrix((num_init, num_init))
    ctr_mass = sparse.kron(sparse.eye(levels, format="csr"), mass, format="csr")
    ctr_stiffness = sparse.kron(
        sparse.eye(levels, format="csr"), stiffness, format="csr"
    )
    if levels > 1:
        rows = levels if prob.ctr.periodic else levels - 1
        diff = sparse.lil_matrix((rows, levels))
        for level in range(levels - 1):
            diff[level, level] = -1.0
            diff[level, level + 1] = 1.0
        if prob.ctr.periodic:
            diff[-1, -1] = -1.0
            diff[-1, 0] = 1.0
        diff = diff.tocsr()
        ctr_time = sparse.kron(diff.T @ diff, mass, format="csr")
    else:
        ctr_time = sparse.csr_matrix((size, size))
    ctr_zeros = sparse.csr_matrix(ctr_mass.shape)
    return (
        sparse.block_diag((init, ctr_zeros), format="csr"),
        sparse.block_diag((zeros, ctr_mass), format="csr"),
        sparse.block_diag((zeros, ctr_stiffness), format="csr"),
        sparse.block_diag((zeros, ctr_time), format="csr"),
    )


def _ctr_mats(prob: Dynamic) -> tuple[Dynamic, Dynamic]:
    surface = prob.model.mesh.boundary(prob.ctr.boundary)
    local = {int(node): idx for idx, node in enumerate(surface.mesh_node_ids)}
    control_nodes = np.asarray(prob.ctr_mesh_node_ids, dtype=np.int64)
    _, first = np.unique(control_nodes, return_index=True)
    control_nodes = control_nodes[np.sort(first)]
    try:
        active = np.array(
            [local[int(node)] for node in control_nodes],
            dtype=np.int64,
        )
    except KeyError as error:
        raise RuntimeError(
            "control node is not part of its boundary surface"
        ) from error

    stiffness, mass, _ = surface.laplacian_matrices()
    stiffness = stiffness[active][:, active].tocsr()
    mass = mass[active][:, active].tocsr()
    if prob.ctr.kind == "vector":
        components = sparse.eye(prob.model.mesh.dimension, format="csr")
        stiffness = sparse.kron(stiffness, components, format="csr")
        mass = sparse.kron(mass, components, format="csr")
    return stiffness, mass


@dataclass(frozen=True)
class Regularization:
    """Navier--Stokes regularization and its individual components."""

    obj: object
    bg: np.ndarray
    bg_obj: object
    space_obj: object
    time_obj: object
    init_bg_obj: object
    ctr_bg_obj: object
    bg_weight: float
    init_weight: float
    space_weight: float
    time_weight: float


def make_reg(
    prob: Dynamic,
    *,
    bg: Dynamic = None,
    bg_weight: float = 0.0,
    init_weight: Dynamic = None,
    space_weight: float = 0.0,
    time_weight: float = 0.0,
) -> Regularization:
    """Build spatial, temporal, and background regularization."""

    bg = prob.param if bg is None else _params(bg, prob.num_param, "bg")
    bg_weight = _weight(bg_weight, "bg_weight")
    init_weight = _weight(
        bg_weight if init_weight is None else init_weight,
        "init_weight",
    )
    space_weight = _weight(space_weight, "space_weight")
    time_weight = _weight(time_weight, "time_weight")

    stiffness, mass = _ctr_mats(prob)
    init_bg_obj = None
    ctr_bg_obj = None
    if prob.num_init_param:
        init_mat, bg_mat, space_mat, time_mat = _combined_mats(
            prob,
            stiffness,
            mass,
        )
        init_bg_obj = _full_reg(prob, init_mat, init_weight, bg)
        ctr_bg_obj = _full_reg(prob, bg_mat, bg_weight, bg)
        bg_obj = femx.SumTimeObjective(
            prob.model.num_steps,
            prob.model.num_states,
            prob.num_param,
        )
        bg_obj.add(init_bg_obj)
        bg_obj.add(ctr_bg_obj)
        space_obj = _full_reg(prob, space_mat, space_weight, bg)
        time_obj = _full_reg(prob, time_mat, time_weight, bg)
    else:
        levels, size = prob.param_shape
        bg_obj = _block_reg(prob, mass, bg_weight, bg)
        space_obj = _block_reg(
            prob,
            stiffness,
            space_weight,
            np.zeros(prob.num_param),
        )
        time_obj = femx.TimeRegularization(
            prob.model.num_steps,
            prob.model.num_states,
            levels,
            size,
            time_weight,
            0.0,
            bg,
        )
        if prob.ctr.periodic and levels > 1:
            _, _, _, time_mat = _combined_mats(prob, stiffness, mass)
            time_obj = _full_reg(prob, time_mat, time_weight, bg)

    obj = femx.SumTimeObjective(
        prob.model.num_steps,
        prob.model.num_states,
        prob.num_param,
    )
    obj.add(bg_obj)
    obj.add(space_obj)
    obj.add(time_obj)

    return Regularization(
        obj=obj,
        bg=bg,
        bg_obj=bg_obj,
        space_obj=space_obj,
        time_obj=time_obj,
        init_bg_obj=init_bg_obj,
        ctr_bg_obj=ctr_bg_obj,
        bg_weight=bg_weight,
        init_weight=init_weight,
        space_weight=space_weight,
        time_weight=time_weight,
    )
