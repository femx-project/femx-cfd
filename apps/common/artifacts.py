"""Write visualization and analysis artifacts."""

import base64
import os
import shutil
import stat
import struct
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np

from femx_cfd._typing import Config, Dynamic, PathLike
from femx_cfd.data import VelocityData


@contextmanager
def _atomic_path(path: PathLike) -> Iterator[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    handle, tmp_name = tempfile.mkstemp(dir=path.parent)
    os.close(handle)
    tmp = Path(tmp_name)
    try:
        yield tmp
        tmp.chmod(mode)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _save_result(
    path: PathLike,
    method: str,
    vals: Config,
    traj: Dynamic,
) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "femx-cfd"
        h5.attrs["format_version"] = 1
        h5.attrs["kind"] = f"{method}_result"
        for name, val in vals.items():
            h5.create_dataset(name, data=val)
        h5.create_dataset("states", data=traj.values, compression="gzip")


def _save_metrics(group: h5py.Group, metrics: Config) -> None:
    for name, val in metrics.items():
        group.attrs[name] = val


def bundle_result(
    src_result: PathLike,
    src_xdmf: PathLike,
    out: PathLike,
    metrics: Optional[Config] = None,
) -> None:
    """Merge femx visualization data into the analysis result HDF5 file."""
    src_result = Path(src_result)
    src_xdmf = Path(src_xdmf)
    out = Path(out)

    xdmf = src_xdmf.read_text(encoding="utf-8")
    heavy_name = next(
        word.split(":", 1)[0]
        for word in xdmf.replace(">", " ").replace("<", " ").split()
        if ".h5:" in word
    )
    heavy_file = src_xdmf.parent / heavy_name

    result_file = out / "result.h5"
    flow_file = out / "flow.xdmf"
    if result_file.exists() or flow_file.exists():
        raise FileExistsError(f"run output already exists: {out}")

    with _atomic_path(result_file) as tmp:
        shutil.copyfile(src_result, tmp)
        with h5py.File(tmp, "a") as result, h5py.File(heavy_file) as flow:
            flow.copy("Mesh", result)
            flow.copy("Data", result)
            if metrics:
                _save_metrics(result.create_group("metrics"), metrics)

    with _atomic_path(flow_file) as tmp:
        tmp.write_text(
            xdmf.replace(f"{heavy_name}:", "result.h5:"),
            encoding="utf-8",
        )


def write_result(
    out: PathLike,
    model: Dynamic,
    method: str,
    vals: Config,
    traj: Dynamic,
    metrics: Config,
) -> None:
    """Write ``result.h5`` and ``flow.xdmf`` for one analysis run."""
    with tempfile.TemporaryDirectory(prefix="femx-cfd-result-") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        result_file = tmp_dir / "result.h5"
        flow_file = tmp_dir / "flow.xdmf"
        _save_result(result_file, method, vals, traj)
        model.write_xdmf(flow_file, traj)
        bundle_result(result_file, flow_file, out, metrics)


def write_truth(path: PathLike, model: Dynamic, traj: Dynamic) -> None:
    """Write the complete synthetic truth trajectory as HDF5/XDMF."""

    model.write_xdmf(Path(path).with_suffix(".xdmf"), traj)


def _vtk_binary(values: np.ndarray, dtype: Dynamic) -> str:
    values = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    payload = values.tobytes()
    header = struct.pack("<Q", len(payload))
    return base64.b64encode(header + payload).decode("ascii")


def _vtk_data(
    parent: ET.Element,
    values: np.ndarray,
    vtk_type: str,
    dtype: Dynamic,
    **attrs: str,
) -> None:
    data = ET.SubElement(
        parent,
        "DataArray",
        type=vtk_type,
        format="binary",
        **attrs,
    )
    data.text = _vtk_binary(values, dtype)


def _write_observation_vtu(
    path: PathLike,
    points: np.ndarray,
    values: np.ndarray,
) -> None:
    num_points, dimension = points.shape
    points_3d = np.zeros((num_points, 3))
    values_3d = np.zeros((num_points, 3))
    points_3d[:, :dimension] = points
    values_3d[:, :dimension] = values

    root = ET.Element(
        "VTKFile",
        type="UnstructuredGrid",
        version="1.0",
        byte_order="LittleEndian",
        header_type="UInt64",
    )
    grid = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(
        grid,
        "Piece",
        NumberOfPoints=str(num_points),
        NumberOfCells=str(num_points),
    )
    point_data = ET.SubElement(piece, "PointData", Vectors="velocity")
    _vtk_data(
        point_data,
        values_3d,
        "Float64",
        "<f8",
        Name="velocity",
        NumberOfComponents="3",
    )
    ET.SubElement(piece, "CellData")

    vtk_points = ET.SubElement(piece, "Points")
    _vtk_data(
        vtk_points,
        points_3d,
        "Float64",
        "<f8",
        NumberOfComponents="3",
    )

    cells = ET.SubElement(piece, "Cells")
    _vtk_data(
        cells,
        np.arange(num_points),
        "Int64",
        "<i8",
        Name="connectivity",
    )
    _vtk_data(
        cells,
        np.arange(1, num_points + 1),
        "Int64",
        "<i8",
        Name="offsets",
    )
    _vtk_data(
        cells,
        np.ones(num_points),
        "UInt8",
        "u1",
        Name="types",
    )

    ET.indent(root, space="  ")
    with _atomic_path(path) as tmp:
        ET.ElementTree(root).write(
            tmp,
            encoding="utf-8",
            xml_declaration=True,
        )


def write_observations(path: PathLike, velocity: VelocityData) -> Path:
    """Write binary VTU point clouds and their ParaView time-series index."""
    heavy_file = Path(path)
    pvd_file = heavy_file.with_name("observations.pvd")
    data_dir = heavy_file.with_name("observations")

    with h5py.File(heavy_file, "a") as h5:
        expected_shape = velocity.vals.shape
        if h5["velocity/values"].shape != expected_shape:
            raise ValueError("velocity HDF5 data has an incompatible shape")

    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True)

    root = ET.Element(
        "VTKFile",
        type="Collection",
        version="1.0",
        byte_order="LittleEndian",
    )
    collection = ET.SubElement(root, "Collection")
    for step, time in enumerate(velocity.times):
        name = f"observation_{step:05d}.vtu"
        _write_observation_vtu(
            data_dir / name,
            velocity.points,
            velocity.vals[step],
        )
        ET.SubElement(
            collection,
            "DataSet",
            timestep=f"{time:.17g}",
            group="",
            part="0",
            file=f"observations/{name}",
        )

    ET.indent(root, space="  ")
    with _atomic_path(pvd_file) as tmp:
        ET.ElementTree(root).write(
            tmp,
            encoding="utf-8",
            xml_declaration=True,
        )

    return pvd_file
