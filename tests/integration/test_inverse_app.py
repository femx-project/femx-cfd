"""End-to-end tests for portable Var and EnVar case execution."""

import json
from pathlib import Path

import h5py
import pytest
import yaml

from femx_cfd.data import load_inlet
from femx_cfd.runtime.runner import run_case


EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "inverse" / "straight_tube"


@pytest.mark.parametrize("method", ("var", "envar"))
def test_runs_portable_inverse_case(method: str, tmp_path: Path) -> None:
    output = run_case(
        EXAMPLE / method / "case.yaml",
        tmp_path / method,
        progress_kind="console",
    )

    status = json.loads((output / "status.json").read_text())
    manifest = json.loads((output / "result.json").read_text())
    assert status["format"] == "femx-cfd-inverse-status"
    assert status["state"] == "complete"
    assert manifest["format"] == "femx-cfd-inverse-result"
    assert manifest["method"] == method
    assert (output / manifest["flow"]).is_file()
    assert (output / manifest["initial_inlet"]).is_file()
    with h5py.File(output / manifest["data"]) as result:
        assert result.attrs["kind"] == f"{method}_result"
        assert "metrics" in result
        assert "Data" in result


def test_derives_initial_inlet_from_observations(tmp_path: Path) -> None:
    document = yaml.safe_load((EXAMPLE / "var" / "case.yaml").read_text())
    document["mesh"]["path"] = str(EXAMPLE / "mesh.msh")
    document["observations"]["path"] = str(EXAMPLE / "observations.h5")
    document["initialization"]["inlet"] = {
        "source": "observations",
        "projection": {
            "depth": 0.5,
            "axial_decay": 0.25,
            "laplacian_modes": 3,
            "ridge": 0.001,
        },
    }
    case_path = tmp_path / "case.yaml"
    case_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    output = run_case(case_path, tmp_path / "result", progress_kind="console")

    inlet = load_inlet(output / "initial_inlet.h5")
    assert inlet.source == "velocity"
    assert inlet.boundary == "inlet"
    assert inlet.vals.ndim == 3
    with h5py.File(output / "initial_inlet.h5") as h5:
        assert h5.attrs["format_version"] == 2
        assert "velocity" in h5["inlet"]
        assert "normal" not in h5["inlet"]
