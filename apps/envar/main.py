"""Run EnVar with prepared velocity observations."""

import os

import hydra
from omegaconf import DictConfig

import femx_cfd as inv
from apps.common import artifacts
from apps.common import config
from apps.common import parallel
from apps.common import progress
from femx_cfd.models import navier_stokes as ns


def run(case, out, prog, runtime):
    opts = case["envar"]
    inlet = inv.load_inlet(case["inlet_file"])

    prog.stage("ensemble: building the initial-state basis")
    spin = ns.build_ensemble(case, inlet, prog.stage)

    prog.stage("input: loading prepared velocity")
    velocity = inv.load_velocity(case["velocity"])

    prob = ns.make_problem(
        case,
        times=inlet.times,
        periodic=inlet.periodic,
        normal=inlet.normal,
        init_mean=spin.init_mean,
        init_modes=spin.init_modes,
    )

    obs = inv.make_velocity_obs(velocity, prob.model)
    param_basis = ns.make_param_basis(
        prob,
        spin.ctr_mean,
        spin.ctr_modes,
    )

    prog.stage("ensemble: building the observation basis")
    obs_basis = ns.build_obs_basis(
        case,
        inlet,
        spin,
        velocity,
        param_basis,
        prog.stage,
    )

    prog.stage("EnVar: estimating ensemble coefficients")
    envar = inv.EnVar(
        prob,
        obs,
        param_basis,
        obs_basis=obs_basis,
        basis_forwards=param_basis.num_coefficients + 1,
        prior_weight=opts["prior_weight"],
        backend=case["solver"]["backend"],
    )
    res = envar.solve(prog=prog)

    if not runtime.is_root:
        return

    prog.stage("output: writing result.h5 and flow.xdmf")
    if opts.get("overwrite", False):
        (out / "result.h5").unlink(missing_ok=True)
        (out / "flow.xdmf").unlink(missing_ok=True)
    metrics = {
        "backend": case["solver"]["backend"],
        "num_modes": prob.num_init_param,
        "linear_objective": res.obj,
        "linear_observation_misfit": res.obs_misfit,
        "linear_prior_penalty": res.prior,
        "linear_gradient_norm": res.grad_norm,
        "nonlinear_observation_misfit": res.nonlinear_misfit,
        "forward_solves": res.num_forwards,
    }

    vals = {
        "param": res.param,
        "coeff": res.coeff,
        "pred": res.pred,
        "linear_pred": res.linear_pred,
    }

    artifacts.write_result(
        out,
        prob.model,
        "envar",
        vals,
        res.traj,
        metrics,
    )


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="synthetic/envar",
)
def main(cfg: DictConfig):
    case = config.to_dict(cfg.case)
    out = config.analysis_output(case)

    petsc_opts = case["solver"].get("petsc_options")
    if petsc_opts:
        os.environ.setdefault("PETSC_OPTIONS", petsc_opts)

    runtime = parallel.for_case(case)
    if runtime.is_root:
        config.save_config(cfg, out / ".hydra" / "config.yaml")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    with progress.console(runtime.is_root) as prog:
        run(case, out, prog, runtime)

    runtime.barrier()
    if runtime.is_root:
        print(f"complete: {out}")
    runtime.finalize()


if __name__ == "__main__":
    main()
