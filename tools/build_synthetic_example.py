"""Generate synthetic mesh, observations, and inlet data from YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

from femx_cfd.runtime.progress import ConsoleProgress, JsonProgress
from synthetic.case import build_from_file


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "examples" / "inverse" / "straight_tube" / "synthetic.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "examples" / "inverse" / "straight_tube",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress", choices=("console", "json"), default="console")
    args = parser.parse_args()
    progress = ConsoleProgress() if args.progress == "console" else JsonProgress()
    with progress:
        build_from_file(
            args.config,
            args.output,
            force=args.force,
            progress=progress,
        )


if __name__ == "__main__":
    main()
