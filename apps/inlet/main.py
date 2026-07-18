"""Prepare inlet data from a waveform or discrete velocity samples."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

import femx_cfd as inv
from apps.common import config
from apps.common import progress
from femx_cfd._typing import Config, Progress
from femx_cfd.prep import inlet as inlet_prep


def run(case: Config, prog: Progress) -> None:
    source = case["inlet"]["source"]
    prog.stage(f"inlet: preparing {source} boundary data")

    inlet = inlet_prep.make_inlet(case)

    path = Path(case["inlet"]["output"])
    path.parent.mkdir(parents=True, exist_ok=True)

    inv.save_inlet(path, inlet)
    prog.stage(f"output: writing {path}")


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="synthetic/truth_inlet",
)
def main(cfg: DictConfig) -> None:
    case = config.to_dict(cfg.case)
    out = Path(case["inlet"]["output"]).parent

    out.mkdir(parents=True, exist_ok=True)

    name = Path(case["inlet"]["output"]).stem
    config.save_config(cfg, out / f"{name}.yaml")

    with progress.ConsoleProgress() as prog:
        run(case, prog)

    print(f"complete: {case['inlet']['output']}")


if __name__ == "__main__":
    main()
