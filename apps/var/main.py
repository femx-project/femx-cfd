"""Run Var with prepared velocity observations."""

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
    backend = case["solver"]["backend"]
    inlet = inv.load_inlet(case["inlet_file"])

    prog.stage("spin-up: computing the initial velocity")
    prob_opts = {
        "times": inlet.times,
        "periodic": inlet.periodic,
        "normal": inlet.normal,
    }

    spin_prob = ns.make_problem(case, **prob_opts)
    bg = inlet.params_for(spin_prob)
    spin = spin_prob.create_solver(backend).solve(bg, progress=prog)

    prog.stage("input: loading prepared velocity")
    velocity = inv.load_velocity(case["velocity"])

    prob = ns.make_problem(case, init_state=spin.values[-1], **prob_opts)
    obs = inv.make_velocity_obs(velocity, prob.model)

    prog.stage("Var: optimizing the inlet velocity")
    opts = case["var"]
    reg_opts = opts["regularization"]

    reg = ns.make_reg(
        prob,
        bg=bg,
        bg_weight=reg_opts["background"],
        space_weight=reg_opts["space"],
        time_weight=reg_opts["time"],
    )

    var = inv.Var(
        prob,
        obs,
        reg=reg.obj,
        init_params=bg,
        backend=backend,
    )

    bnds = [tuple(opts["control_bounds"])] * prob.num_param

    res = var.solve(
        bg,
        opt=opts["optimizer"],
        bnds=bnds,
        max_itrs=opts["max_iterations"],
        grad_tol=opts["gradient_tolerance"],
        func_tol=opts["function_tolerance"],
        prog=prog,
    )

    if not runtime.is_root:
        return

    prog.stage("output: writing result.h5 and flow.xdmf")
    if opts.get("overwrite", False):
        (out / "result.h5").unlink(missing_ok=True)
        (out / "flow.xdmf").unlink(missing_ok=True)
    metrics = {
        "backend": backend,
        "optimizer": opts["optimizer"],
        "objective": res.obj,
        "observation_misfit": res.obs_misfit,
        "regularization": res.reg,
        "gradient_norm": res.grad_norm,
        "iterations": res.num_iter,
        "converged": res.converged,
        "message": res.msg,
    }

    vals = {
        "param": res.param,
        "pred": res.pred,
    }

    artifacts.write_result(
        out,
        prob.model,
        "var",
        vals,
        res.traj,
        metrics,
    )


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="synthetic/var",
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

    with progress.console(runtime.is_root) as prog:
        run(case, out, prog, runtime)

    runtime.barrier()
    if runtime.is_root:
        print(f"complete: {out}")
    runtime.finalize()


if __name__ == "__main__":
    main()
