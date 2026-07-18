"""Generate mesh inputs used by the inverse-analysis workflows."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from apps.common import config
from apps.common import progress
from femx_cfd._typing import Config, Progress
from femx_cfd.mesh import write_straight_tube


def run(case: Config, prog: Progress) -> Path:
    opts = case["mesh"]
    if opts["generator"] != "straight_tube_2d":
        raise ValueError("mesh generator must be 'straight_tube_2d'")

    output = Path(opts["output"])
    if output.exists() and not opts.get("overwrite", False):
        raise FileExistsError(
            f"mesh already exists: {output}; set case.mesh.overwrite=true to replace it"
        )

    prog.stage(f"mesh: generating {opts['generator']}")
    write_straight_tube(
        output,
        length=opts["length"],
        diameter=opts["diameter"],
        cell_size=opts["cell_size"],
    )
    prog.stage(f"output: writing {output}")
    return output


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="synthetic/mesh",
)
def main(cfg: DictConfig) -> None:
    case = config.to_dict(cfg.case)
    with progress.ConsoleProgress() as prog:
        output = run(case, prog)
    print(f"complete: {output}")


if __name__ == "__main__":
    main()
