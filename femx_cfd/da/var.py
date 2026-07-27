"""Variational data-assimilation method."""

from dataclasses import dataclass

import femx
import numpy as np
from scipy.optimize import minimize

from .._typing import Dynamic, OptionalProgress
from ..models.navier_stokes.solver import (
    platform_name,
    select_solver,
    solver_name,
)


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


@dataclass(frozen=True)
class VarResult:
    """Result of a variational solve and its final forward validation."""

    param: np.ndarray
    traj: object
    pred: np.ndarray
    obj: float
    obs_misfit: float
    reg: float
    grad_norm: float
    num_iter: int
    num_fun: int
    num_grad: int
    converged: bool
    status: int
    msg: str


class Var:
    """Minimize an observation objective with a discrete adjoint."""

    def __init__(
        self,
        prob: Dynamic,
        obs: Dynamic,
        *,
        reg: Dynamic = None,
        init_params: Dynamic = None,
        platform: Dynamic = "cpu",
        backend: Dynamic = "dense",
        solver_options: Dynamic = None,
    ) -> None:
        if obs.op.model is not prob.model:
            raise ValueError("prob and obs must use the same model")
        if prob.num_param == 0:
            raise ValueError("prob must contain parameters")

        self._prob = prob
        self._obs = obs
        self._reg = reg
        self._init_params = (
            prob.param
            if init_params is None
            else _params(
                init_params,
                prob.num_param,
                "init_params",
            )
        )
        self._obs_obj = obs.objective(num_param=prob.num_param)
        if reg is None:
            self._obj = self._obs_obj
        else:
            self._obj = femx.SumTimeObjective(
                prob.model.num_steps,
                prob.model.num_states,
                prob.num_param,
            )
            self._obj.add(self._obs_obj)
            self._obj.add(reg)
        memory_space, solver_type = select_solver(platform, backend)
        self._reduced = prob.create_reduced(
            self._obj,
            memory_space,
            solver_type,
            options=solver_options,
        )

    @property
    def prob(self) -> Dynamic:
        return self._prob

    @property
    def obs(self) -> Dynamic:
        return self._obs

    @property
    def reg(self) -> Dynamic:
        return self._reg

    @property
    def init_params(self) -> np.ndarray:
        return self._init_params.copy()

    @property
    def obs_obj(self) -> Dynamic:
        return self._obs_obj

    @property
    def obj(self) -> Dynamic:
        return self._obj

    @property
    def reduced(self) -> Dynamic:
        return self._reduced

    @property
    def backend(self) -> str:
        return solver_name(self._reduced.solver_type)

    @property
    def solver_type(self) -> str:
        return solver_name(self._reduced.solver_type)

    @property
    def platform(self) -> str:
        return platform_name(self._reduced.memspace)

    def value_grad(
        self,
        params: np.ndarray,
        prog: OptionalProgress = None,
    ) -> tuple[float, np.ndarray]:
        return self._reduced.value_grad(params, progress=prog)

    def solve(
        self,
        init_params: Dynamic = None,
        *,
        opt: str = "scipy",
        bnds: Dynamic = None,
        max_itrs: int = 100,
        grad_tol: float = 1.0e-8,
        func_tol: float = 1.0e-12,
        prog: OptionalProgress = None,
    ) -> VarResult:
        """Solve with SciPy L-BFGS-B or PETSc/TAO LMVM."""

        if opt not in ("scipy", "tao"):
            raise ValueError("opt must be 'scipy' or 'tao'")
        if isinstance(max_itrs, bool) or not isinstance(max_itrs, (int, np.integer)):
            raise TypeError("max_itrs must be an integer")
        if max_itrs <= 0:
            raise ValueError("max_itrs must be positive")
        if prog is not None and not callable(prog):
            raise TypeError("prog must be callable")
        grad_tol = _weight(grad_tol, "grad_tol")
        if opt == "scipy":
            func_tol = _weight(func_tol, "func_tol")
        init_params = (
            self._init_params
            if init_params is None
            else _params(
                init_params,
                self._prob.num_param,
                "init_params",
            )
        )

        if opt == "scipy":
            num_eval = 0

            def evaluate(params: np.ndarray) -> tuple[float, np.ndarray]:
                nonlocal num_eval
                obj, grad = self.value_grad(params, prog=prog)
                num_eval += 1
                if prog is not None:
                    prog(
                        {
                            "type": "optimizer",
                            "iteration": num_eval,
                            "max_iterations": max_itrs,
                            "iteration_kind": "evaluation",
                            "objective": obj,
                            "gradient_norm": float(np.linalg.norm(grad)),
                            "param": params.copy(),
                        }
                    )
                return obj, grad

            res = minimize(
                evaluate,
                init_params,
                jac=True,
                method="L-BFGS-B",
                bounds=bnds,
                options={
                    "maxiter": int(max_itrs),
                    "gtol": grad_tol,
                    "ftol": func_tol,
                },
            )
            params = np.asarray(res.x, dtype=float).copy()
            grad = np.asarray(res.jac, dtype=float)
            num_iter = int(res.nit)
            num_fun = int(res.nfev)
            num_grad = int(res.njev)
            converged = bool(res.success)
            status = int(res.status)
            msg = str(res.message)
        else:
            res = femx.TaoOptimizer(self._reduced).solve(
                init_params,
                bnds=bnds,
                max_itrs=max_itrs,
                grad_tol=grad_tol,
                progress=prog,
            )
            params = res.param
            grad = res.grad
            num_iter = res.num_iter
            num_fun = res.num_fun
            num_grad = res.num_grad
            converged = res.converged
            status = res.status
            msg = res.msg

        if prog is not None:
            prog(
                {
                    "type": "stage",
                    "name": "validation",
                    "message": "final forward solve",
                }
            )
        traj = self._prob.solve(
            params,
            self._reduced.memspace,
            self._reduced.solver_type,
        )
        pred = self._obs.op.predict(traj)
        obs_misfit = float(self._obs_obj.value(traj, params))
        reg = 0.0 if self._reg is None else float(self._reg.value(traj, params))
        return VarResult(
            param=params,
            traj=traj,
            pred=np.asarray(pred, dtype=float).copy(),
            obj=obs_misfit + reg,
            obs_misfit=obs_misfit,
            reg=reg,
            grad_norm=float(np.linalg.norm(grad)),
            num_iter=num_iter,
            num_fun=num_fun,
            num_grad=num_grad,
            converged=converged,
            status=status,
            msg=msg,
        )
