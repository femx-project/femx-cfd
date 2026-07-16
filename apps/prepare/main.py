"""Prepare velocity data from a synthetic solve or MRI."""

import os
from pathlib import Path

import femx
import hydra
from omegaconf import DictConfig

import femx_cfd as inv
from apps.common import artifacts
from apps.common import config
from apps.common import parallel
from apps.common import progress
from femx_cfd.prep import mri
from femx_cfd.prep import synthetic


def prepare_synthetic(case, prog, runtime):
    inlet = inv.load_inlet(case["input"])
    truth_prob, truth_params, cycles = synthetic.make_truth_problem(
        case,
        inlet,
    )

    end_time = case["simulation"]["end_time"]

    prog.stage(f"forward: solving synthetic truth until t={end_time}")
    solver = truth_prob.create_solver(case["solver"]["backend"])
    truth = solver.solve(truth_params, progress=prog)

    if not runtime.is_root:
        return

    prog.stage("output: writing truth.h5 and truth.xdmf")
    artifacts.write_truth(case["truth"], truth_prob.model, truth)

    prog.stage("downsampling: sampling the final cycle")

    pts = synthetic.obs_points(case)
    times = synthetic.obs_times(case)
    truth_op = femx.VelocityObservationOperator(
        truth_prob.model,
        pts,
        synthetic.obs_times(case, cycle=cycles),
        components=range(truth_prob.model.mesh.dimension),
    )

    vals = truth_op.predict(truth)

    velocity = inv.VelocityData(
        points=pts,
        times=times,
        vals=truth_op.reshape(vals),
    )

    prog.stage("output: writing velocity.h5 and observations.pvd")
    inv.save_velocity(case["velocity"], velocity)
    artifacts.write_observations(case["velocity"], velocity)

    for name, opts in case.get("noise", {}).items():
        prog.stage(f"noise: writing {name} observations")
        noisy = inv.add_velocity_noise(
            velocity,
            std=opts["std"],
            seed=opts["seed"],
        )
        inv.save_velocity(opts["output"], noisy)


def prepare_mri(case, prog, runtime):
    if not runtime.is_root:
        return
    prog.stage("input: reading MRI velocity data")
    pts, times, vals = mri.load_mri(case["input"])

    velocity = inv.VelocityData(points=pts, times=times, vals=vals)

    prog.stage("output: writing velocity.h5 and observations.pvd")
    inv.save_velocity(case["velocity"], velocity)
    artifacts.write_observations(case["velocity"], velocity)


def run(case, prog, runtime):
    if case["source"] == "synthetic":
        prepare_synthetic(case, prog, runtime)
    elif case["source"] == "mri":
        prepare_mri(case, prog, runtime)
    else:
        raise ValueError("prepare source must be 'synthetic' or 'mri'")


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="synthetic/prepare",
)
def main(cfg: DictConfig):
    case = config.to_dict(cfg.case)
    out = Path(case["prepared"])

    petsc_opts = case.get("solver", {}).get("petsc_options")
    if petsc_opts:
        os.environ.setdefault("PETSC_OPTIONS", petsc_opts)

    runtime = parallel.for_case(case)
    if runtime.is_root:
        out.mkdir(parents=True, exist_ok=True)
        config.save_config(cfg, out / "prepare.yaml")

    with progress.console(runtime.is_root) as prog:
        run(case, prog, runtime)

    runtime.barrier()
    if runtime.is_root:
        print(f"complete: {out}")
    runtime.finalize()


if __name__ == "__main__":
    main()
