"""Resolve case mesh units without modifying source mesh files."""

from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Iterator

import gmsh

from femx_cfd._typing import Config
from femx_cfd.cases.validation import LENGTH_SCALES


@contextmanager
def mesh_in_meters(mesh_options: Config) -> Iterator[Path]:
    """Yield a meter-scaled mesh path for one validated case."""

    path = Path(mesh_options["path"])
    scale = LENGTH_SCALES[mesh_options["length_unit"]]
    if scale == 1.0:
        yield path
        return
    if gmsh.isInitialized():
        raise RuntimeError("mesh scaling requires ownership of the Gmsh runtime")

    with tempfile.TemporaryDirectory(prefix="femx-cfd-mesh-") as directory:
        scaled = Path(directory) / "mesh.msh"
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(str(path))
            gmsh.model.mesh.affineTransform(
                [
                    scale,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    scale,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    scale,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            )
            gmsh.write(str(scaled))
        finally:
            gmsh.finalize()
        yield scaled


__all__ = ["mesh_in_meters"]
