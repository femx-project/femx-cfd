"""Ensemble variational data-assimilation method."""

from dataclasses import dataclass
from typing import Callable, Optional

import femx
import numpy as np
from scipy.optimize import lsq_linear

from .._typing import Dynamic, OptionalProgress
from ..models.navier_stokes.solver import (
    platform_name,
    select_solver,
    solver_name,
)


def _bound_val(val: Dynamic, default: float) -> float:
    if val is None:
        return default
    if isinstance(val, (bool, np.bool_)):
        raise TypeError("coefficient bounds must contain real values or None")
    try:
        out = float(val)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "coefficient bounds must contain real values or None"
        ) from error
    if np.isnan(out):
        raise ValueError("coefficient bounds must not contain NaN")
    return out


def _bound_pair(vals: Dynamic) -> tuple[float, float]:
    try:
        pair = tuple(vals)
    except TypeError as error:
        raise TypeError(
            "each coefficient bound must be a (lower, upper) pair"
        ) from error
    if len(pair) != 2:
        raise ValueError("each coefficient bound must be a (lower, upper) pair")
    return _bound_val(pair[0], -np.inf), _bound_val(pair[1], np.inf)


def _is_bound_val(val: Dynamic) -> bool:
    return val is None or (np.isscalar(val) and not isinstance(val, (str, bytes)))


def _coeff_bounds(
    bounds: Dynamic,
    size: int,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if bounds is None:
        return None
    try:
        entries = tuple(bounds)
    except TypeError as error:
        raise TypeError(
            "bounds must be a (lower, upper) pair or one pair per coefficient"
        ) from error

    if len(entries) == 2 and all(_is_bound_val(val) for val in entries):
        lower_val, upper_val = _bound_pair(entries)
        lower = np.full(size, lower_val)
        upper = np.full(size, upper_val)
    else:
        if len(entries) != size:
            raise ValueError(f"bounds must contain {size} coefficient pairs")
        pairs = [_bound_pair(entry) for entry in entries]
        lower = np.asarray([pair[0] for pair in pairs])
        upper = np.asarray([pair[1] for pair in pairs])

    if np.any(lower >= upper):
        raise ValueError(
            "each coefficient lower bound must be less than its upper bound"
        )
    return np.ascontiguousarray(lower), np.ascontiguousarray(upper)


def _pred(
    predict: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
) -> np.ndarray:
    vals = np.asarray(predict(params.copy()), dtype=float)
    if vals.ndim != 1 or vals.size == 0:
        raise ValueError("predict must return a nonempty one-dimensional array")
    if not np.all(np.isfinite(vals)):
        raise ValueError("predict returned non-finite observations")
    return np.ascontiguousarray(vals)


def _notify(
    prog: OptionalProgress,
    phase: str,
    status: str,
    step: int,
    total: int,
) -> None:
    if prog is not None:
        prog(
            {
                "type": "ensemble",
                "phase": phase,
                "status": status,
                "step": step,
                "total": total,
            }
        )


def _build_obs_basis(
    param_basis: Dynamic,
    predict: Callable[[np.ndarray], np.ndarray],
    prog: OptionalProgress = None,
) -> Dynamic:
    if not isinstance(param_basis, femx.EnsembleBasis):
        raise TypeError("param_basis must be an EnsembleBasis")
    if not callable(predict):
        raise TypeError("predict must be callable")
    if prog is not None and not callable(prog):
        raise TypeError("prog must be callable")

    total = param_basis.num_coefficients + 1
    mean_params = param_basis.mean
    _notify(prog, "mean", "begin", 1, total)

    mean_pred = _pred(predict, mean_params)
    _notify(prog, "mean", "end", 1, total)

    obs_modes = np.empty(
        (mean_pred.size, param_basis.num_coefficients),
        dtype=float,
    )

    param_modes = param_basis.perturbations
    for col in range(param_basis.num_coefficients):
        step = col + 2
        _notify(prog, "member", "begin", step, total)
        pred = _pred(predict, mean_params + param_modes[:, col])
        if pred.size != mean_pred.size:
            raise ValueError("predict returned inconsistent observation sizes")
        obs_modes[:, col] = pred - mean_pred
        _notify(prog, "member", "end", step, total)

    return femx.EnsembleBasis(mean_pred, obs_modes)


@dataclass(frozen=True)
class _LinearResult:
    coeff: np.ndarray
    coefficient_covariance: np.ndarray
    param: np.ndarray
    pred: np.ndarray
    obj: float
    obs_misfit: float
    prior: float
    grad_norm: float


class _LinearProblem:
    """Solve the linear observation-space problem used inside EnVar."""

    def __init__(
        self,
        param_basis: Dynamic,
        obs_basis: Dynamic,
        *,
        prior_weight: float,
    ) -> None:
        if not isinstance(param_basis, femx.EnsembleBasis):
            raise TypeError("param_basis must be an EnsembleBasis")
        if not isinstance(obs_basis, femx.EnsembleBasis):
            raise TypeError("obs_basis must be an EnsembleBasis")
        if obs_basis.num_coefficients != param_basis.num_coefficients:
            raise ValueError(
                "parameter and observation bases must use the same coefficients"
            )
        if not np.isfinite(prior_weight) or prior_weight < 0.0:
            raise ValueError("prior_weight must be finite and nonnegative")

        self._param_basis = param_basis
        self._obs_basis = obs_basis
        self._prior_weight = float(prior_weight)

    def solve(
        self,
        obs: Dynamic,
        std: Dynamic = 1.0,
        bounds: Dynamic = None,
    ) -> _LinearResult:
        obs = np.asarray(obs, dtype=float)
        if obs.ndim != 1 or obs.size != self._obs_basis.value_size:
            raise ValueError("obs must match the observation basis value size")
        if not np.all(np.isfinite(obs)):
            raise ValueError("obs must be finite")

        std = np.asarray(std, dtype=float)

        try:
            std = np.broadcast_to(std, obs.shape)
        except ValueError as error:
            raise ValueError("std must be scalar or match obs") from error
        if not np.all(np.isfinite(std)) or np.any(std <= 0.0):
            raise ValueError("std must be finite and positive")

        obs_mean = self._obs_basis.mean
        modes = self._obs_basis.perturbations / std[:, None]
        innovation = (obs - obs_mean) / std
        num_coeff = self._param_basis.num_coefficients

        lhs = modes
        rhs = innovation

        if self._prior_weight > 0.0:
            scale = np.sqrt(self._prior_weight)
            lhs = np.vstack((lhs, scale * np.eye(num_coeff)))
            rhs = np.concatenate((rhs, np.zeros(num_coeff)))
        limits = _coeff_bounds(bounds, num_coeff)
        if limits is None:
            coeff = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        else:
            res = lsq_linear(lhs, rhs, bounds=limits)
            if not res.success:
                raise RuntimeError(
                    f"bounded EnVar least-squares solve failed: {res.message}"
                )
            coeff = res.x
        inverse_design = np.linalg.pinv(lhs)
        coefficient_covariance = inverse_design @ inverse_design.T

        params = self._param_basis.evaluate(coeff)
        pred = self._obs_basis.evaluate(coeff)
        residual = (pred - obs) / std
        obs_misfit = 0.5 * float(residual @ residual)
        prior = 0.5 * self._prior_weight * float(coeff @ coeff)
        grad = modes.T @ residual + self._prior_weight * coeff

        return _LinearResult(
            coeff=coeff.copy(),
            coefficient_covariance=coefficient_covariance,
            param=params.copy(),
            pred=pred.copy(),
            obj=obs_misfit + prior,
            obs_misfit=obs_misfit,
            prior=prior,
            grad_norm=float(np.linalg.norm(grad)),
        )


@dataclass(frozen=True)
class EnVarResult:
    """Result of an EnVar solve and its nonlinear forward validation."""

    coeff: np.ndarray
    coefficient_covariance: np.ndarray
    param: np.ndarray
    traj: object
    pred: np.ndarray
    linear_pred: np.ndarray
    obj: float
    obs_misfit: float
    prior: float
    grad_norm: float
    nonlinear_misfit: float
    num_forwards: int


class EnVar:
    """Estimate parameters in an ensemble subspace."""

    def __init__(
        self,
        prob: Dynamic,
        obs: Dynamic,
        param_basis: Dynamic,
        *,
        obs_basis: Dynamic = None,
        basis_forwards: int = 0,
        prior_weight: float = 1.0,
        platform: Dynamic = "cpu",
        backend: Dynamic = "dense",
        solver_options: Dynamic = None,
    ) -> None:
        if not isinstance(param_basis, femx.EnsembleBasis):
            raise TypeError("param_basis must be an EnsembleBasis")
        if obs.op.model is not prob.model:
            raise ValueError("prob and obs must use the same model")
        if param_basis.value_size != prob.num_param:
            raise ValueError("param_basis size must match prob.param")
        if not np.isfinite(prior_weight) or prior_weight < 0.0:
            raise ValueError("prior_weight must be finite and nonnegative")
        if obs_basis is not None:
            if not isinstance(obs_basis, femx.EnsembleBasis):
                raise TypeError("obs_basis must be an EnsembleBasis")
            if obs_basis.num_coefficients != param_basis.num_coefficients:
                raise ValueError(
                    "parameter and observation bases must use the same coefficients"
                )
            if obs_basis.value_size != obs.op.num_obs:
                raise ValueError("obs_basis size must match observations")
        if not isinstance(basis_forwards, int) or basis_forwards < 0:
            raise ValueError("basis_forwards must be a nonnegative integer")
        if obs_basis is None and basis_forwards:
            raise ValueError("basis_forwards requires a precomputed obs_basis")

        self._prob = prob
        self._obs = obs
        self._param_basis = param_basis
        self._prior_weight = float(prior_weight)
        self._obs_basis = obs_basis
        self._basis_forwards = basis_forwards
        memory_space, solver_type = select_solver(platform, backend)
        self._solver = prob.create_solver(
            memory_space,
            solver_type,
            options=solver_options,
        )
        self._obj = obs.objective(num_param=prob.num_param)

    @property
    def prob(self) -> Dynamic:
        return self._prob

    @property
    def obs(self) -> Dynamic:
        return self._obs

    @property
    def param_basis(self) -> Dynamic:
        return self._param_basis

    @property
    def obs_basis(self) -> Dynamic:
        return self._obs_basis

    @property
    def solver(self) -> Dynamic:
        return self._solver

    @property
    def backend(self) -> str:
        return solver_name(self._solver.solver_type)

    @property
    def solver_type(self) -> str:
        return solver_name(self._solver.solver_type)

    @property
    def platform(self) -> str:
        return platform_name(self._solver.memspace)

    @property
    def obs_obj(self) -> Dynamic:
        return self._obj

    @property
    def num_forwards(self) -> int:
        return self._basis_forwards + self._solver.num_solves

    def _predict(self, params: np.ndarray) -> np.ndarray:
        return self._obs.op.predict(self._solver.solve(params))

    def build_obs_basis(
        self,
        force: bool = False,
        prog: OptionalProgress = None,
    ) -> Dynamic:
        if prog is not None and not callable(prog):
            raise TypeError("prog must be callable")
        if self._obs_basis is None or force:
            if force:
                self._basis_forwards = 0
            self._obs_basis = _build_obs_basis(
                self._param_basis,
                self._predict,
                prog=prog,
            )
        return self._obs_basis

    def solve(
        self,
        prog: OptionalProgress = None,
        bounds: Dynamic = None,
    ) -> EnVarResult:
        lin = _LinearProblem(
            self._param_basis,
            self.build_obs_basis(prog=prog),
            prior_weight=self._prior_weight,
        ).solve(self._obs.values, self._obs.std, bounds=bounds)

        if prog is not None:
            prog(
                {
                    "type": "stage",
                    "message": "EnVar: forward solve at analysis",
                }
            )
        traj = self._solver.solve(lin.param, progress=prog)
        pred = self._obs.op.predict(traj)
        return EnVarResult(
            coeff=lin.coeff,
            coefficient_covariance=lin.coefficient_covariance,
            param=lin.param,
            traj=traj,
            pred=pred.copy(),
            linear_pred=lin.pred,
            obj=lin.obj,
            obs_misfit=lin.obs_misfit,
            prior=lin.prior,
            grad_norm=lin.grad_norm,
            nonlinear_misfit=self._obj.value(traj, lin.param),
            num_forwards=self.num_forwards,
        )
