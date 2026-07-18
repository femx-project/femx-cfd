"""Mesh generation for the synthetic inverse example."""

from __future__ import annotations

import math
from pathlib import Path
from types import ModuleType

from femx_cfd._typing import PathLike


def _require_gmsh() -> ModuleType:
    try:
        import gmsh
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "mesh generation requires Gmsh; install it with `pip install gmsh`"
        ) from error
    return gmsh


def _boundary_curves(
    gmsh: ModuleType,
    length: float,
    surface: int,
) -> tuple[list[int], list[int], list[int]]:
    """Return the rectangle's inlet, outlet, and wall curve tags."""

    half_length = length / 2
    tolerance = max(length, 1.0) * 1e-10
    inlet, outlet, wall = [], [], []

    for _, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        xmin, _, _, xmax, _, _ = gmsh.model.getBoundingBox(1, tag)
        center_x = (xmin + xmax) / 2
        if math.isclose(center_x, -half_length, abs_tol=tolerance):
            inlet.append(tag)
        elif math.isclose(center_x, half_length, abs_tol=tolerance):
            outlet.append(tag)
        else:
            wall.append(tag)

    if len(inlet) != 1 or len(outlet) != 1 or len(wall) != 2:
        raise RuntimeError("could not identify straight-tube boundary curves")
    return inlet, outlet, wall


def _normalize(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
    )


def write_straight_tube(
    path: PathLike,
    *,
    length: float,
    diameter: float,
    cell_size: float,
) -> None:
    """Write a 2D rectangular flow mesh with named physical groups."""

    if length <= 0 or diameter <= 0 or cell_size <= 0:
        raise ValueError("length, diameter, and cell_size must all be positive")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gmsh = _require_gmsh()
    owns_session = not gmsh.isInitialized()
    if owns_session:
        gmsh.initialize()

    created_model = False
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.add("straight_tube")
        created_model = True

        surface = gmsh.model.occ.addRectangle(
            -length / 2,
            -diameter / 2,
            0,
            length,
            diameter,
        )
        gmsh.model.occ.synchronize()

        inlet, outlet, wall = _boundary_curves(gmsh, length, surface)
        groups = (
            (2, [surface], 3, "fluid"),
            (1, inlet, 4, "inlet"),
            (1, outlet, 5, "outlet"),
            (1, wall, 6, "wall"),
        )
        for dimension, entities, tag, name in groups:
            gmsh.model.addPhysicalGroup(dimension, entities, tag)
            gmsh.model.setPhysicalName(dimension, tag, name)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), cell_size)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
        _normalize(path)
    finally:
        if created_model:
            gmsh.model.remove()
        if owns_session:
            gmsh.finalize()
