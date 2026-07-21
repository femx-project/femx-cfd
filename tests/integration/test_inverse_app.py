"""End-to-end tests for portable Var and EnVar case execution."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
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
    assert (output / manifest["flow_window"]).is_file()
    assert (output / manifest["flow_observation"]).is_file()
    assert (output / manifest["observation"]).is_file()
    assert (output / manifest["initial_inlet"]).is_file()
    with h5py.File(output / manifest["data"]) as result:
        assert result.attrs["kind"] == f"{method}_result"
        assert "metrics" in result
        assert "Data" in result
        if method == "envar":
            assert result["param"].shape == (28,)
            assert result["coeff"].shape == (4,)
            assert result["param_history"].shape == (2, 28)
            assert result["coeff_history"].shape == (2, 4)
            assert result["metrics"].attrs["outer_iterations"] == 2
            assert result["metrics"].attrs[
                "nonlinear_observation_misfit_history"
            ].shape == (2,)
        else:
            assert result["param"].shape == (24,)
        assert "SampleData" not in result
    phase_times = [
        float(element.attrib["Value"])
        for element in ET.parse(output / manifest["flow_observation"]).findall(
            "./Domain/Grid/Grid/Time"
        )
    ]
    with h5py.File(EXAMPLE / "observation" / "velocity.h5") as observation:
        assert phase_times == pytest.approx(observation["velocity/times"][...])
    pvd = (output / manifest["observation"]).read_text(encoding="utf-8")
    assert 'file="velocity/velocity_00000.vtu"' in pvd


def test_derives_initial_inlet_from_observations(tmp_path: Path) -> None:
    document = yaml.safe_load((EXAMPLE / "var" / "case.yaml").read_text())
    document["mesh"]["path"] = str(EXAMPLE / "mesh.msh")
    document["observations"]["path"] = str(EXAMPLE / "observation" / "velocity.h5")
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

    inlet = load_inlet(output / "inlet" / "initial_velocity.h5")
    assert inlet.source == "velocity"
    assert inlet.boundary == "inlet"
    assert inlet.vals.ndim == 3
    with h5py.File(output / "inlet" / "initial_velocity.h5") as h5:
        assert h5.attrs["format_version"] == 2
        assert "velocity" in h5["inlet"]
        assert "normal" not in h5["inlet"]


def test_writes_envar_initial_ensemble_views(tmp_path: Path) -> None:
    document = yaml.safe_load((EXAMPLE / "envar" / "case.yaml").read_text())
    document["mesh"]["path"] = str(EXAMPLE / "mesh.msh")
    document["observations"]["path"] = str(
        EXAMPLE / "observation" / "velocity.h5"
    )
    document["initialization"]["inlet"]["path"] = str(
        EXAMPLE / "inlet" / "initial_velocity.h5"
    )
    document["ensemble"]["save_initial_states"] = True
    case_path = tmp_path / "envar-initial-ensemble.yaml"
    case_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    output = run_case(case_path, tmp_path / "result", progress_kind="console")

    manifest = json.loads((output / "result.json").read_text())
    views = manifest["ensemble_initial_states"]
    assert views == [
        "estimate/ensemble_initial_outer_01.xdmf",
        "estimate/ensemble_initial_outer_02.xdmf",
    ]
    with h5py.File(output / "estimate" / "result.h5") as result:
        outer = result["EnsembleInitial/Outer000"]
        labels = outer["labels"].asstr()[...].tolist()
        assert labels[0] == "mean"
        assert labels[-1] == "analysis"
        assert len(labels) == 6
        assert labels[1].startswith("alpha_")
        assert not np.allclose(
            outer["Member000/velocity"],
            outer["Member001/velocity"],
        )
    names = [
        grid.attrib["Name"]
        for grid in ET.parse(output / views[0]).findall("./Domain/Grid/Grid")
    ]
    assert names == labels


def test_runs_optional_normal_control(tmp_path: Path) -> None:
    document = yaml.safe_load((EXAMPLE / "var" / "case.yaml").read_text())
    document["mesh"]["path"] = str(EXAMPLE / "mesh.msh")
    document["observations"]["path"] = str(EXAMPLE / "observation" / "velocity.h5")
    document["initialization"]["inlet"]["path"] = str(
        EXAMPLE / "inlet" / "initial_velocity.h5"
    )
    document["control"] = {
        "boundary": "inlet",
        "kind": "normal",
        "normal": "auto",
    }
    document["algorithm"]["control_bounds"] = [0.0, 0.2]
    case_path = tmp_path / "normal.yaml"
    case_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    output = run_case(case_path, tmp_path / "normal", progress_kind="console")

    with h5py.File(output / "estimate" / "result.h5") as result:
        assert result["param"].shape == (12,)
